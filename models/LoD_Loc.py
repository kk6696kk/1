# maploc/models/LoD_Loc_learnable.py

"""
LoD-Loc with Learnable Refinement
带可学习refinement的LoD-Loc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel
from .LDConv4 import FeatExtNet
from .learnable_refinement import DifferentiablePoseRefinement
from .utils import (
    point_proj, 
    sample_poses, 
    find_max, 
    pose2euler,
    norm_uv,
    transf
)
from .voting import (
    log_softmax_spatial,
    softmax_spatial,
    get_score,
    get_mean_score,
)
from .metrics import (
    AngleError, AngleRecall, 
    Location2DError_x, Location2DError_y, Location2DError_z, Location2DError, 
    Location2DRecall_xy, Location2DRecall_z, Location2DRecall,
    AllRecall, AllRecall_6DoF, RError
)


class LoD_Loc(BaseModel):
    """
    LoD-Loc with end-to-end learnable refinement
    """
    default_conf = {
        "num_sample": "???",
        "num_sample_val": "???",
        "error_ranges": "???",
        "lamb": "???",
        "stage_configs": "???",
        "lamb_val": "???",
        "feat_ext_ch": "???",
        "loss_weight": "???",
        "confidence": False,
        "loss_id": "softmax",
        
        # Learnable refinement配置
        'learnable_refinement': {
            'enable': True,
            'start_level': 1,  # 从stage 2开始
            'max_iterations': 5,
            'use_confidence_gating': True,  # 是否使用置信度门控
            'confidence_threshold': 0.3,  # 置信度阈值
        },
        
        # Loss权重
        'loss_weights': {
            'confidence': 0.1,  # 置信度loss权重
            'pose_consistency': 0.05,  # Pose一致性loss
            'feature_quality': 0.02,  # 特征质量loss
        }
    }

    def _init(self, conf):
        self.stage_configs = self.conf.stage_configs
        self.lamb = self.conf.lamb
        self.lamb_val = self.conf.lamb_val
        self.error_ranges = self.conf.error_ranges
        self.num_sample = self.conf.num_sample
        self.num_sample_val = self.conf.num_sample_val
        self.num_stage = len(self.conf.stage_configs)
        self.loss_weight = self.conf.loss_weight
        self.loss_id = self.conf.loss_id
        self.confidence = self.conf.confidence

        # Feature extraction
        self.feature_extraction = FeatExtNet(
            base_channels=self.conf.feat_ext_ch, 
            num_stage=self.num_stage
        )
        
        # Learnable refinement module
        if self.conf.learnable_refinement['enable']:
            self.learnable_refiner = DifferentiablePoseRefinement(
                num_levels=self.num_stage,
                max_iterations=self.conf.learnable_refinement['max_iterations'],
                feature_dim=self.conf.feat_ext_ch
            )
            print(f"✓ Learnable refinement enabled from level {self.conf.learnable_refinement['start_level']}")
        else:
            self.learnable_refiner = None
    
    def _forward(self, data):
        pred = {}
        features = self.feature_extraction(data["image"])
        exp_var, pred_pose = None, None
        lamb_ = None
        
        pred_pose_, rxyz_pred_ = None, None

        # Multi-stage localization
        for stage_idx in range(self.num_stage):
            output = {}
            
            # 选择lambda
            if data['epoch_stage'] == 'train':
                lamb_ = self.lamb[stage_idx]
            elif data['epoch_stage'] == 'val':
                lamb_ = self.lamb_val[stage_idx]
            
            # 获取特征
            features_stage = features[f"stage{stage_idx + 1}_f"]
            conf_feature = features[f"stage{stage_idx + 1}_c"]

            if self.confidence:
                f_weight = features_stage * conf_feature
            else:  
                f_weight = features_stage
            
            # ===== Pose采样 =====
            if stage_idx == 0:
                ranges = torch.tensor(self.error_ranges)
                output["ranges"] = ranges
                if data['epoch_stage'] == 'train':
                    output["num_sample"] = torch.tensor(self.num_sample[stage_idx])
                elif data['epoch_stage'] == 'val':
                    output["num_sample"] = torch.tensor(self.num_sample_val[stage_idx])
                poses_init = data['pose_sample'] @ transf.to(data['pose_sample'])
            else:
                low_bound = -exp_var
                high_bound = exp_var
                ranges = torch.stack((low_bound, high_bound), dim=1).permute(0, 2, 1)
                if data['epoch_stage'] == 'train':
                    output["num_sample"] = torch.tensor(self.num_sample[stage_idx])
                elif data['epoch_stage'] == 'val':
                    output["num_sample"] = torch.tensor(self.num_sample_val[stage_idx])
                output["ranges"] = ranges
                poses_init = pred_pose @ transf.to(pred_pose)
            
            # ===== Pose投影与打分 =====
            poses_sampled, rxyz_sampled, output["sample_euler"], PitchRoll = sample_poses(
                poses_init[:, 0:3, 3], 
                poses_init[:, 0:3, 0:3], 
                ranges, 
                output["num_sample"], 
                data['pose_GT'].squeeze()
            )
            
            uv_sampled = point_proj(
                data["points3D"], 
                poses_sampled, 
                data['intrinsic'], 
                data['origin_hw']
            )

            _, _, new_h, new_w = f_weight.shape
            scale_factors = torch.stack([
                (new_h-1) / data['origin_hw'][:, 0], 
                (new_w-1) / data['origin_hw'][:, 1]
            ], dim=1)
            uv_sampled[:, :, :, 0] *= scale_factors[0, 1]
            uv_sampled[:, :, :, 1] *= scale_factors[0, 0]
            
            uv_sampled = norm_uv(uv_sampled, new_h, new_w)
            score = get_score(f_weight, uv_sampled)
            del uv_sampled

            score_mean = get_mean_score(score)
            output["prob"] = softmax_spatial(score_mean, dims=1)
            
            # ===== 初始Pose选择 =====
            if self.loss_id == "softmax" or self.loss_id == "kl_loss":
                output["log_prob"] = log_softmax_spatial(score_mean, dims=1)
                output["w_feature"] = f_weight
                output["pred_score_mean"], output["pred_pose"], output["pred_score"] = find_max(
                    output["log_prob"], poses_sampled, score
                )
                with torch.no_grad():
                    xyz_pred, euler_pred = pose2euler(output["pred_pose"])
                    output["rxyz_pred"] = torch.hstack([
                        euler_pred[..., 2].unsqueeze(1), xyz_pred
                    ])
            
            del score_mean
            del score

            # ===== Learnable Refinement =====
            if (self.learnable_refiner is not None and 
                stage_idx >= self.conf.learnable_refinement['start_level']):
                
                # ===== 新增：决定使用哪个特征 =====
                # 最后一个stage使用最精细的stage_fine特征
                if stage_idx == self.num_stage - 1 and 'stage_fine' in features:
                    refine_features = features['stage_fine']
                    use_fine_feat = True
                    print(f"[Stage {stage_idx+1}] Using stage_fine features")
                else:
                    refine_features = f_weight
                    use_fine_feat = False
                
                # 获取特征图尺寸
                _, _, refine_h, refine_w = refine_features.shape
                
                # ===== 新增：调整内参以匹配特征图尺寸 =====
                if use_fine_feat:
                    # Fine特征可能有不同的尺寸，需要重新计算内参
                    intrinsic = data['intrinsic'].clone()
                    scale_h = (refine_h - 1) / data['origin_hw'][:, 0]
                    scale_w = (refine_w - 1) / data['origin_hw'][:, 1]
                    
                    intrinsic[:, 0, 0] *= scale_w  # fx
                    intrinsic[:, 1, 1] *= scale_h  # fy
                    intrinsic[:, 0, 2] *= scale_w  # cx
                    intrinsic[:, 1, 2] *= scale_h  # cy
                else:
                    # 使用原有的scale_factors逻辑
                    _, _, new_h, new_w = f_weight.shape
                    scale_factors = torch.stack([
                        (new_h-1) / data['origin_hw'][:, 0], 
                        (new_w-1) / data['origin_hw'][:, 1]
                    ], dim=1)
                    
                    intrinsic = data['intrinsic'].clone()
                    intrinsic[:, 0, 0] *= scale_factors[:, 1]
                    intrinsic[:, 1, 1] *= scale_factors[:, 0]
                    intrinsic[:, 0, 2] *= scale_factors[:, 1]
                    intrinsic[:, 1, 2] *= scale_factors[:, 0]
                
                # 保存refinement前的pose
                output["pred_pose_before_refine"] = output["pred_pose"].clone()
                
                # 应用learnable refinement
                refined_pose, refine_aux = self.learnable_refiner(
                    pose=output["pred_pose"],
                    features=refine_features,  # ← 使用合适的特征
                    points_3d=data["points3D"],
                    intrinsics=intrinsic,
                    image_size=(refine_h, refine_w),
                    level=stage_idx,
                    prob_distribution=output["prob"],
                    training=self.training
                )
                
                # 可选：根据置信度阈值决定是否使用refined pose
                if self.conf.learnable_refinement['use_confidence_gating']:
                    confidence_threshold = self.conf.learnable_refinement['confidence_threshold']
                    use_refined = (refine_aux['confidence'] > confidence_threshold).float().view(-1, 1, 1)
                    output["pred_pose"] = use_refined * refined_pose + (1 - use_refined) * output["pred_pose"]
                else:
                    output["pred_pose"] = refined_pose
                
                # 保存refinement信息
                output["refine_aux"] = refine_aux
                output["used_fine_features"] = use_fine_feat
                
                # 重新计算rxyz_pred
                with torch.no_grad():
                    xyz_pred, euler_pred = pose2euler(output["pred_pose"])
                    output["rxyz_pred"] = torch.hstack([
                        euler_pred[..., 2].unsqueeze(1), xyz_pred
                    ])
            
            # ===== 计算方差 =====
            samp_variance = (rxyz_sampled - output["rxyz_pred"].unsqueeze(1)) ** 2
            output["exp_variance"] = lamb_ * (
                torch.sum(
                    samp_variance * output["prob"].unsqueeze(2).expand(-1, -1, 4), 
                    dim=1, 
                    keepdim=False
                ) ** 0.5
            )
            
            # 更新全局变量
            exp_var = output["exp_variance"]
            pred_pose = output["pred_pose"]
            pred[f"stage{stage_idx + 1}"] = output
            pred_pose_, rxyz_pred_ = output["pred_pose"], output["rxyz_pred"]
        
        return {
            **pred,
            "rxyz_pred": rxyz_pred_,
            "pred_pose": pred_pose_,
            "fine_feat": features.get('stage_fine', None),
        }

    def loss(self, pred, data):
        """
        计算总loss，包括原有loss和refinement相关loss
        """
        # 1. 原有的定位loss
        if self.loss_id == "softmax":
            from .utils import multi_stage_loss
            nll = multi_stage_loss(pred, data['pose_GT'][..., 2:], self.loss_weight, self.num_stage)
        elif self.loss_id == "kl_loss":
            from .utils import multi_stage_loss_KL
            nll = multi_stage_loss_KL(pred, data, self.loss_weight, self.num_stage)
        elif self.loss_id == "l1_loss":
            from .utils import multi_stage_loss_l1loss
            nll = multi_stage_loss_l1loss(pred, data['pose_GT'][..., 2:], self.loss_weight, self.num_stage)
        else:
            raise ValueError(f"Unknown loss_id: {self.loss_id}")
        
        loss_dict = {"total": nll, "nll": nll.clone()}
        
        # 2. Refinement相关的loss
        if self.learnable_refiner is not None:
            refine_losses = self._compute_refinement_losses(pred, data)
            
            for key, value in refine_losses.items():
                loss_dict[key] = value
                loss_dict["total"] += value
        
        return loss_dict
    
    def _compute_refinement_losses(self, pred, data):
        """
        计算refinement相关的辅助loss
        """
        losses = {}
        gt_pose = data['pose_GT_4x4'].float()
        
        total_confidence_loss = 0
        total_consistency_loss = 0
        total_feature_loss = 0
        num_refined_stages = 0
        
        for stage_idx in range(self.conf.learnable_refinement['start_level'], self.num_stage):
            stage_key = f"stage{stage_idx + 1}"
            
            if stage_key not in pred or "refine_aux" not in pred[stage_key]:
                continue
            
            stage_output = pred[stage_key]
            refine_aux = stage_output["refine_aux"]
            
            num_refined_stages += 1
            
            # ===== 1. Confidence Supervision Loss =====
            # 目标：当refinement后pose更接近GT时，confidence应该高
            
            # 计算refinement前后的pose误差
            pose_before = stage_output["pred_pose_before_refine"]
            pose_after = stage_output["pred_pose"]
            
            # Translation error
            t_error_before = torch.norm(
                pose_before[:, :3, 3] - gt_pose[:, :3, 3], 
                dim=-1
            )  # [B]
            t_error_after = torch.norm(
                pose_after[:, :3, 3] - gt_pose[:, :3, 3], 
                dim=-1
            )  # [B]
            
            # 如果refinement后误差更小，target confidence应该高
            error_reduction = t_error_before - t_error_after  # 正值表示改善
            
            # Target: sigmoid映射，改善越多confidence越高
            target_confidence = torch.sigmoid(error_reduction)  # [B]
            
            predicted_confidence = refine_aux['confidence'].squeeze(-1)  # [B]
            
            confidence_loss = F.mse_loss(predicted_confidence, target_confidence)
            total_confidence_loss += confidence_loss
            
            # ===== 2. Pose Consistency Loss =====
            # 鼓励refinement不要改变太多（正则化）
            
            pose_diff = torch.norm(
                pose_after[:, :3, 3] - pose_before[:, :3, 3], 
                dim=-1
            ).mean()
            
            # 使用Huber loss：允许小的改变，惩罚大的改变
            consistency_loss = F.smooth_l1_loss(
                pose_after[:, :3, 3], 
                pose_before[:, :3, 3],
                beta=0.5
            )
            total_consistency_loss += consistency_loss
            
            # ===== 3. Feature Quality Loss =====
            # 鼓励refinement features在正确投影位置有高响应
            
            if refine_aux.get('final_score') is not None:
                # 计算GT pose下的feature响应作为上界
                with torch.no_grad():
                    gt_score, _ = self.learnable_refiner.compute_alignment_score(
                        gt_pose,
                        data["points3D"],
                        refine_aux['refine_features'],
                        data['intrinsic'],
                        refine_aux['refine_features'].shape[-2:]
                    )
                
                current_score = refine_aux['final_score']
                
                # 鼓励当前score接近GT score
                feature_loss = F.mse_loss(current_score, gt_score.detach())
                total_feature_loss += feature_loss
        
        # 平均并加权
        if num_refined_stages > 0:
            losses['refine_confidence_loss'] = (
                total_confidence_loss / num_refined_stages * 
                self.conf.loss_weights['confidence']
            )
            losses['refine_consistency_loss'] = (
                total_consistency_loss / num_refined_stages * 
                self.conf.loss_weights['pose_consistency']
            )
            losses['refine_feature_loss'] = (
                total_feature_loss / num_refined_stages * 
                self.conf.loss_weights['feature_quality']
            )
        
        return losses

    def metrics(self):
        """定义评估指标"""
        return {
            "xyz_error": Location2DError("rxyz_pred"),
            "x_error": Location2DError_x("rxyz_pred"),
            "y_error": Location2DError_y("rxyz_pred"),
            "z_error": Location2DError_z("rxyz_pred"),
            "yaw_error": AngleError("rxyz_pred"),
            "R_error": RError("pred_pose"),

            "xy_recall_2dot5m": Location2DRecall_xy(0.25, "rxyz_pred"),
            "xy_recall_1m": Location2DRecall_xy(1.0, "rxyz_pred"),
            "xy_recall_2m": Location2DRecall_xy(2.0, "rxyz_pred"),
            "xy_recall_3m": Location2DRecall_xy(3.0, "rxyz_pred"),
            "xy_recall_5m": Location2DRecall_xy(5.0, "rxyz_pred"),
            "xy_recall_10m": Location2DRecall_xy(10.0, "rxyz_pred"),
            "xy_recall_20m": Location2DRecall_xy(20.0, "rxyz_pred"),

            "z_recall_2dot5m": Location2DRecall_z(0.25, "rxyz_pred"),
            "z_recall_1m": Location2DRecall_z(1.0, "rxyz_pred"),
            "z_recall_2m": Location2DRecall_z(2.0, "rxyz_pred"),
            "z_recall_3m": Location2DRecall_z(3.0, "rxyz_pred"),
            "z_recall_5m": Location2DRecall_z(5.0, "rxyz_pred"),
            "z_recall_10m": Location2DRecall_z(10.0, "rxyz_pred"),
            "z_recall_20m": Location2DRecall_z(20.0, "rxyz_pred"),

            "xyz_recall_2dot5m": Location2DRecall(0.25, "rxyz_pred"),
            "xyz_recall_1m": Location2DRecall(1.0, "rxyz_pred"),
            "xyz_recall_2m": Location2DRecall(2.0, "rxyz_pred"),
            "xyz_recall_3m": Location2DRecall(3.0, "rxyz_pred"),
            "xyz_recall_5m": Location2DRecall(5.0, "rxyz_pred"),
            "xyz_recall_10m": Location2DRecall(10.0, "rxyz_pred"),
            "xyz_recall_20m": Location2DRecall(20.0, "rxyz_pred"),

            "yaw_recall_1°": AngleRecall(1.0, "rxyz_pred"),
            "yaw_recall_2°": AngleRecall(2.0, "rxyz_pred"),
            "yaw_recall_3°": AngleRecall(3.0, "rxyz_pred"),
            "yaw_recall_5°": AngleRecall(5.0, "rxyz_pred"),
            "yaw_recall_7°": AngleRecall(7.0, "rxyz_pred"),
            "yaw_recall_10°": AngleRecall(10.0, "rxyz_pred"),

            "AllRecall6DoF_1m1°": AllRecall_6DoF(1.0, 1.0, "pred_pose"),
            "AllRecall6DoF_2m2°": AllRecall_6DoF(2.0, 2.0, "pred_pose"),
            "AllRecall6DoF_3m3°": AllRecall_6DoF(3.0, 3.0, "pred_pose"),
            "AllRecall6DoF_5m5°": AllRecall_6DoF(5.0, 5.0, "pred_pose"),
            "AllRecall6DoF_10m7°": AllRecall_6DoF(10.0, 7.0, "pred_pose"),
            "AllRecall6DoF_20m10°": AllRecall_6DoF(20.0, 10.0, "pred_pose")
        }