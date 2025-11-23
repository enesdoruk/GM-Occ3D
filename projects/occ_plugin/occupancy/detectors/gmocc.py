import torch
import collections 
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from .bevdepth import BEVDepth
from mmdet3d.models import builder
from mmengine.registry import MODELS


import numpy as np
import time
import copy

from projects.occ_plugin.utils import SPConvVoxelization

import importlib
importlib.import_module('projects.occ_plugin.occupancy.lifter.gaussian_lifter')
importlib.import_module('projects.occ_plugin.occupancy.lifter.gaussian_lifter_v2')
importlib.import_module('projects.occ_plugin.occupancy.backbones.gaussian_encoder.anchor_encoder_module')
importlib.import_module('projects.occ_plugin.occupancy.backbones.gaussian_encoder.deformable_module')


@DETECTORS.register_module()
class GMOcc(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            gauss_lifter=None,
            gauss_encoder=None,
            head=None,
            dataset='nuscenes',
            loss_norm=False,
            **kwargs):     
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None

        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.dataset = dataset
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        
        self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg) if pts_voxel_cfg is not None else None
        
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone) if occ_encoder_backbone is not None else None
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck) if occ_encoder_neck is not None else None

        self.gauss_lifter = MODELS.build(gauss_lifter) if gauss_lifter is not None else None

        self.gauss_encoder = MODELS.build(gauss_encoder) if gauss_encoder is not None else None
        
        self.head = MODELS.build(head) if head is not None else None
        

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
    
        x = list(x)
        for i in range(len(x)):            
            _, output_dim, ouput_H, output_W = x[i].shape
            x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
        x = tuple(x)
        
        return {'img_feats': x}
    
    @force_fp32()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    
    def extract_img_feat(self, img, img_metas):        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        img_enc_feats = self.image_encoder(img[0])
        img_feats = img_enc_feats['img_feats']
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)
        
        return img_feats

    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        xyz = coors[:, [3, 2, 1]].unsqueeze(0).float().contiguous()
        batch_size = coors[-1, 0] + 1
        
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, pts=xyz, batch_size=batch_size)
        pts_enc_feats['pts_feats'] = pts_enc_feats['pts_feats'][0].permute(0,2,1,3,4)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        return pts_enc_feats['pts_feats']

    def extract_feat(self, points, img, img_metas):
        pts_feats = None
        img_feats =  None
        if img is not None:
            img_feats = self.extract_img_feat(img, img_metas)
        if points is not None:
            pts_feats = self.extract_pts_feat(points)        

        cam_positions= torch.stack([torch.tensor(t['cam_positions']).to(points[0].device) for t in img_metas], dim=0)
        focal_positions= torch.stack([torch.tensor(t['focal_positions']).to(points[0].device) for t in img_metas], dim=0)
        projection_mat= torch.stack([torch.tensor(t['projection_mat']).to(points[0].device) for t in img_metas], dim=0)
        occ_cam_mask= torch.stack([torch.tensor(t['occ_cam_mask']).to(points[0].device) for t in img_metas], dim=0)
        occ_label= torch.stack([torch.tensor(t['occ_label']).to(points[0].device) for t in img_metas], dim=0)
        occ_xyz= torch.stack([torch.tensor(t['occ_xyz']).to(points[0].device) for t in img_metas], dim=0)

        img_shapes = [tuple(img_feats[0].shape[-2:]) for _ in range(img[0].shape[1])]
        image_wh = torch.tensor(np.ascontiguousarray(np.array(img_shapes, dtype=np.float32)[:, :2][:, ::-1])).to(points[0].device).unsqueeze(0)
        
        gauss_metas = dict(cam_positions= cam_positions,
                          focal_positions= focal_positions,
                          projection_mat= projection_mat,
                          occ_cam_mask= occ_cam_mask,
                          occ_label= occ_label,
                          occ_xyz= occ_xyz,
                          secondfpn_out=img_feats[0],
                          image_wh=image_wh)


        lifter = self.gauss_lifter(gauss_metas)
        
        gauss_feats = self.gauss_encoder(lifter['representation'], 
                                        lifter['rep_features'], 
                                        img_feats, gauss_metas)

        return (gauss_metas, gauss_feats, img_feats, pts_feats, lifter)
    
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            **kwargs,
        ):
 
        gauss_metas, gauss_feats, _, _, _ = self.extract_feat(points, img_inputs, img_metas)
          
        output = self.head(gauss_feats['representation'], gauss_metas)
        
        loss = self.head.loss_gauss(output)['loss']
        losses = dict(loss = loss)
        
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
        
        if self.record_time:
            logging_latencies()
             
        return losses
        
    def forward_test(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        
        return self.simple_test(img_metas, img_inputs, points, gt_occ=gt_occ, visible_mask=visible_mask, **kwargs)
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None):
        
        start_time = time.time()

        gauss_metas, gauss_feats, _, _, lifter_metas = self.extract_feat(points, img, img_metas)

        output = self.head(gauss_feats['representation'], gauss_metas)

        end_time = time.time()
        
        result_dict = dict()
        result_dict.update(gauss_metas)
        result_dict.update(gauss_feats)  
        result_dict.update(rep_features=lifter_metas['rep_features'])
        result_dict.update(rep_features=lifter_metas['anchor_init'])  
        result_dict.update(output)
  
        ious,occ_iou, miou, sc_metric, ssc_metric, ssc_occ_metric = self.evaluation_semantic(result_dict)
       
        test_output = {
            'IOU': ious,
            'occ_IOU': occ_iou,
            'mIOU': miou,
            'SC': sc_metric,
            'SSC': ssc_metric,
            'SSC_occ': ssc_occ_metric,
            'time_use': end_time - start_time,
        }

        return test_output


    def evaluation_semantic(self, result_dict):
        class_indices = list(range(1, 17))
        num_classes = len(class_indices)
        empty_label = 17
        label_str = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation']
            
        # compute iou, precision, recall
        for idx, pred in enumerate(result_dict['final_occ']):
            output = pred
            target = result_dict['sampled_label'][idx]
            mask = result_dict['occ_mask'][idx].flatten()
            
            if mask is not None:
                output = output[mask]
                target = target[mask]

            total_seen = torch.zeros(num_classes+1)
            total_correct = torch.zeros(num_classes+1)
            total_positive = torch.zeros(num_classes+1)
            
            for i, c in enumerate(class_indices):
                total_seen[i] += torch.sum(target == c).item()
                total_correct[i] += torch.sum((target == c)
                                                & (output == c)).item()
                total_positive[i] += torch.sum(output == c).item()
            
            total_seen[-1] += torch.sum(target != empty_label).item()
            total_correct[-1] += torch.sum((target != empty_label)
                                                & (output != empty_label)).item()
            total_positive[-1] += torch.sum(output != empty_label).item()
        
        ious = []
        precs = []
        recas = [] 
        for i in range(num_classes):
            if total_positive[i] == 0:
                precs.append(0.)
            else:
                cur_prec = total_correct[i] / total_positive[i]
                precs.append(cur_prec.item())
            if total_seen[i] == 0:
                ious.append(1)
                recas.append(1)
            else:
                cur_iou = total_correct[i] / (total_seen[i]
                                                   + total_positive[i]
                                                   - total_correct[i])
                cur_reca = total_correct[i] / total_seen[i]
                ious.append(cur_iou.item())
                recas.append(cur_reca)
        miou = np.mean(ious)
                
        occ_iou = total_correct[-1] / (total_seen[-1]
                                            + total_positive[-1]
                                            - total_correct[-1])
                
        #SC metric
        pred = result_dict['final_occ'][0]
        gt = result_dict['sampled_label'][0]
        mask = result_dict['occ_mask'][0].flatten()
        valid_pred = pred[mask]
        valid_gt = gt[mask]
        pred_binary = torch.zeros_like(valid_pred)
        gt_binary = torch.zeros_like(valid_gt)
        pred_binary[valid_pred != empty_label] = 1
        gt_binary[valid_gt != empty_label] = 1
        
        gt_binary = gt_binary.cpu().numpy().astype(np.int)
        pred_binary = pred_binary.cpu().numpy()
        noise_mask = gt_binary != 255
        
        sc_metric = fast_hist(pred_binary[noise_mask], gt_binary[noise_mask], max_label=2)
        
        #SSC metric
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy().astype(np.int)
        if mask is not None:
            mask = mask.cpu().numpy()
            noise_mask = gt != 255
            mask = noise_mask & (mask!=0)
            ssc_occ_metric = fast_hist(pred[mask], gt[mask], max_label=num_classes+1)

        ssc_metric = fast_hist(pred[noise_mask], gt[noise_mask], max_label=num_classes+1)
        
        return ious, occ_iou, miou, sc_metric, ssc_metric, ssc_occ_metric
        
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):

        gauss_metas, gauss_feats, img_feats, pts_feats = self.extract_feat(points, img_inputs, img_metas)

        output = self.head(gauss_feats['representation'], gauss_metas)

        return output
    
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
