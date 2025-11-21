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
importlib.import_module('projects.occ_plugin.occupancy.backbones.gaussian_encoder.anchor_encoder_module')


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
      
        gauss_metas = dict(cam_positions= cam_positions,
                          focal_positions= focal_positions,
                          projection_mat= projection_mat,
                          occ_cam_mask= occ_cam_mask,
                          occ_label= occ_label,
                          occ_xyz= occ_xyz)
        
        lifter = self.gauss_lifter(img_feats)
        
        gauss_feats = self.gauss_encoder(lifter['representation'], 
                                        lifter['rep_features'], 
                                        img_feats, gauss_metas)

        return (gauss_metas, gauss_feats, img_feats, pts_feats)
    
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            **kwargs,
        ):

        gauss_metas, gauss_feats, img_feats, pts_feats = self.extract_feat(points, img_inputs, img_metas)
                    
        output = self.head(gauss_feats['representation'], gauss_metas)
                
        losses = dict()
        losses['loss'] = output['loss']
                
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

        gauss_metas, gauss_feats, img_feats, pts_feats = self.extract_feat(points, img, img_metas)

        output = self.head(gauss_feats['representation'], gauss_metas)

        end_time = time.time()

        pred_c = output['output_voxels'][0]
        SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        pred_f = None
        SSC_metric_fine = None
        if output['output_voxels_fine'] is not None:
            if output['output_coords_fine'] is not None:
                fine_pred = output['output_voxels_fine'][0]  # N ncls
                fine_coord = output['output_coords_fine'][0]  # 3 N
                pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
            else:
                pred_f = output['output_voxels_fine'][0]
            SC_metric, _ = self.evaluation_semantic(pred_f, gt_occ, eval_type='SC', visible_mask=visible_mask)
            SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
            'time_use': end_time - start_time,
        }

        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine

        return test_output


    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        _, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None

        if self.dataset == 'kitti':
            max_label = 20
        elif self.dataset == 'poss':
            max_label = 12
        else:
            max_label = 17

        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=max_label)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=max_label)
            return hist, hist_occ
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):

        gauss_metas, gauss_feats, img_feats, pts_feats = self.extract_feat(points, img_inputs, img_metas)

        transform = img_inputs[1:8] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        return output
    
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
