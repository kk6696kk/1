# maploc/models/learnable_refinement.py

"""
Learnable End-to-End Pose Refinement Module
可学习的端到端位姿优化模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidencePredictor(nn.Module):
    """
    预测是否应该进行refinement的置信度网络
    基于当前的features、pose和probability distribution预测
    """
    def __init__(self, feature_dim=8):
        super().__init__()
        
        # 输入: feature统计量(3*C) + pose特征(4) + prob特征(2)
        input_dim = feature_dim * 3 + 6
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, pose, prob_distribution):
        """
        Args:
            features: [B, C, H, W] 特征图
            pose: [B, 4, 4] 预测的pose
            prob_distribution: [B, N_samples] 投票概率分布
        
        Returns:
            confidence: [B, 1] 置信度分数 [0, 1]
        """
        B = features.shape[0]
        
        # 1. Feature统计量
        feat_mean = features.mean(dim=(2, 3))  # [B, C]
        feat_std = features.std(dim=(2, 3))    # [B, C]
        feat_max = features.flatten(2).max(dim=2)[0]  # [B, C]
        
        # 2. Pose特征
        t_norm = torch.norm(pose[:, :3, 3], dim=-1, keepdim=True)  # [B, 1]
        R = pose[:, :3, :3]
        # 旋转矩阵的Frobenius norm
        R_norm = torch.norm(R.reshape(B, -1), dim=-1, keepdim=True)  # [B, 1]
        # 旋转矩阵的trace (接近3表示接近单位矩阵)
        trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1, keepdim=True)  # [B, 1]
        # Determinant (应该接近1)
        det = torch.det(R).unsqueeze(-1)  # [B, 1]
        
        # 3. Probability distribution特征
        prob_max = prob_distribution.max(dim=-1, keepdim=True)[0]  # [B, 1]
        prob_entropy = -(prob_distribution * torch.log(prob_distribution + 1e-8)).sum(-1, keepdim=True)  # [B, 1]
        
        # 拼接所有特征
        x = torch.cat([
            feat_mean, feat_std, feat_max,
            t_norm, R_norm, trace, det,
            prob_max, prob_entropy
        ], dim=-1)  # [B, input_dim]
        
        confidence = self.fc(x)  # [B, 1]
        return confidence


class AdaptiveDampingNetwork(nn.Module):
    """
    自适应学习每个参数的damping因子
    """
    def __init__(self, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        
        # 每个level一个damping预测网络
        self.damping_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(6, 32),  # 输入：6-DoF梯度的统计量
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 6),
                nn.Softplus()  # 确保输出正值
            )
            for _ in range(num_levels)
        ])
        
        # 每个level的基础damping (可学习)
        self.base_log_damping = nn.ParameterList([
            nn.Parameter(torch.zeros(6))  # log空间，初始化为1.0
            for _ in range(num_levels)
        ])
    
    def forward(self, level, gradient_stats):
        """
        Args:
            level: int, 当前stage索引
            gradient_stats: [B, 6] 梯度的统计量（通常是绝对值或平方）
        
        Returns:
            damping: [B, 6] 每个参数的damping因子
        """
        # 基础damping (log空间学习更稳定)
        base_damping = torch.exp(self.base_log_damping[level]).unsqueeze(0)  # [1, 6]
        
        # 根据梯度自适应调整
        adaptive_scale = self.damping_nets[level](gradient_stats)  # [B, 6]
        
        damping = base_damping * adaptive_scale  # [B, 6]
        
        # Clamp到合理范围
        damping = torch.clamp(damping, min=1e-4, max=10.0)
        
        return damping


class FeatureRefinementHead(nn.Module):
    """
    专门用于refinement的轻量级特征提取头
    将原始features转换为更适合alignment的单通道heatmap
    
    🆕 支持任意输入通道数（1, 8, 16等）
    """
    def __init__(self, in_channels=8, hidden_channels=32):
        super().__init__()
        
        self.in_channels = in_channels
        
        # 🆕 如果输入是1通道，先扩展通道
        if in_channels == 1:
            self.channel_expander = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            # 跳过 conv1，直接从 conv2 开始
            self.use_expander = True
        else:
            self.channel_expander = None
            self.use_expander = False
            
            # 原有的 conv1（用于多通道输入）
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
        
        # 后续层保持不变（对两种情况都适用）
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.output = nn.Conv2d(hidden_channels // 2, 1, 1)
    
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W] 原始特征（C可以是1或8等）
        
        Returns:
            refined_features: [B, 1, H, W] refinement专用的单通道特征
        """
        # 🆕 根据输入通道数选择路径
        if self.use_expander:
            # 路径1：1通道 → 扩展到hidden_channels
            x = self.channel_expander(features)
        else:
            # 路径2：多通道 → conv1
            x = self.conv1(features)
        
        # 两条路径汇合，后续处理相同
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x


class DifferentiablePoseRefinement(nn.Module):
    """
    完全可微的位姿优化模块
    使用简化的梯度下降方法而不是LM，确保端到端可训练
    """
    def __init__(self, num_levels=3, max_iterations=5, feature_dim=8):
        super().__init__()
        self.max_iterations = max_iterations
        self.num_levels = num_levels
        
        # 1. 置信度预测器
        self.confidence_predictor = ConfidencePredictor(feature_dim)
        
        # 2. 自适应damping网络
        self.damping_network = AdaptiveDampingNetwork(num_levels)
        
        # 3. Refinement专用特征头
        self.feature_head = FeatureRefinementHead(
            in_channels=feature_dim,
            hidden_channels=32
        )
        
        # 4. 学习率调度（每个level可以不同）
        self.lr_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1)
            for _ in range(num_levels)
        ])
        
    def project_points(self, pose, points_3d, intrinsics):
        """
        可微的3D点投影
        
        Args:
            pose: [B, 4, 4] camera-to-world pose
            points_3d: [B, N, 3] 世界坐标系的3D点
            intrinsics: [B, 3, 3] 内参矩阵
        
        Returns:
            points_2d: [B, N, 2] 投影的2D点
            depth: [B, N] 深度
        """
        B, N, _ = points_3d.shape
        
        # 处理intrinsics
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0).expand(B, -1, -1)
        if intrinsics.shape[-1] == 4:
            intrinsics = intrinsics[:, :3, :3]
        intrinsics = intrinsics.to(dtype=pose.dtype)
        
        # 世界坐标 -> 相机坐标
        pose_w2c = torch.inverse(pose)
        points_homo = torch.cat([
            points_3d,
            torch.ones(B, N, 1, device=points_3d.device, dtype=points_3d.dtype)
        ], dim=-1)
        
        points_cam = torch.bmm(
            pose_w2c[:, :3, :],
            points_homo.transpose(1, 2)
        ).transpose(1, 2)  # [B, N, 3]
        
        # 投影到图像平面
        depth = points_cam[..., 2]  # [B, N]
        points_2d_homo = torch.bmm(
            intrinsics,
            points_cam.transpose(1, 2)
        )  # [B, 3, N]
        
        points_2d = points_2d_homo[:, :2, :] / (points_2d_homo[:, 2:3, :] + 1e-8)
        points_2d = points_2d.transpose(1, 2)  # [B, N, 2]
        
        return points_2d, depth
    
    def sample_features(self, features, points_2d, image_size):
        """
        可微的特征采样
        
        Args:
            features: [B, C, H, W] 特征图
            points_2d: [B, N, 2] 2D点坐标
            image_size: (H, W)
        
        Returns:
            sampled: [B, C, N] 采样的特征
            valid_weights: [B, N] 软化的有效性权重
        """
        B, C, H, W = features.shape
        N = points_2d.shape[1]
        
        # 归一化到[-1, 1]
        grid = points_2d.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        grid = grid.view(B, 1, N, 2)
        
        # 可微采样
        sampled = F.grid_sample(
            features, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # [B, C, 1, N]
        
        sampled = sampled.squeeze(2)  # [B, C, N]
        
        # 计算软化的有效性权重 (边界附近平滑衰减)
        interpolation_pad = 4
        
        # 使用sigmoid实现软边界
        u = points_2d[..., 0]  # [B, N]
        v = points_2d[..., 1]
        
        u_valid = torch.sigmoid((u - interpolation_pad)) * \
                  torch.sigmoid((W - interpolation_pad - 1 - u))
        v_valid = torch.sigmoid((v - interpolation_pad)) * \
                  torch.sigmoid((H - interpolation_pad - 1 - v))
        
        valid_weights = u_valid * v_valid  # [B, N]
        
        return sampled, valid_weights
    
    def compute_alignment_score(self, pose, points_3d, features, intrinsics, image_size):
        """
        计算对齐分数（用于优化的目标函数）
        
        Returns:
            score: [B] 对齐分数（越高越好）
            valid_ratio: [B] 有效点的比例
        """
        B = pose.shape[0]
        
        # 投影
        points_2d, depth = self.project_points(pose, points_3d, intrinsics)
        
        # 深度有效性（软化）
        depth_valid = torch.sigmoid((depth - 0.1) * 10)  # [B, N]
        
        # 采样特征
        sampled, spatial_valid = self.sample_features(features, points_2d, image_size)
        
        # 总有效性权重
        valid_weights = depth_valid * spatial_valid  # [B, N]
        
        # 对齐分数：加权平均的特征响应
        # 假设features是alignment heatmap，值越大越好
        if sampled.shape[1] == 1:
            score = (sampled.squeeze(1) * valid_weights).sum(dim=-1) / (valid_weights.sum(dim=-1) + 1e-6)
        else:
            # 多通道：取最大响应
            score = (sampled.max(dim=1)[0] * valid_weights).sum(dim=-1) / (valid_weights.sum(dim=-1) + 1e-6)
        
        valid_ratio = valid_weights.mean(dim=-1)  # [B]
        
        return score, valid_ratio
    
    def rodrigues_rotation(self, axis_angle):
        """
        Rodrigues公式：轴角 -> 旋转矩阵 (可微)
        
        Args:
            axis_angle: [B, 3]
        
        Returns:
            R: [B, 3, 3]
        """
        B = axis_angle.shape[0]
        device = axis_angle.device
        dtype = axis_angle.dtype
        
        angle = torch.norm(axis_angle, dim=-1, keepdim=True)  # [B, 1]
        
        # 处理小角度情况
        small_angle_mask = (angle < 1e-8).squeeze(-1)
        angle_safe = torch.where(angle < 1e-8, torch.ones_like(angle) * 1e-8, angle)
        
        axis = axis_angle / angle_safe  # [B, 3]
        
        # 构造反对称矩阵 K
        K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        # Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        angle_expanded = angle.unsqueeze(-1)  # [B, 1, 1]
        
        K_squared = torch.bmm(K, K)
        R = I + torch.sin(angle_expanded) * K + (1 - torch.cos(angle_expanded)) * K_squared
        
        # 小角度时返回单位矩阵
        R = torch.where(small_angle_mask.view(B, 1, 1), I, R)
        
        return R
    
    def apply_pose_delta(self, pose, delta):
        """
        应用6-DoF增量到pose (可微)
        
        Args:
            pose: [B, 4, 4]
            delta: [B, 6] [delta_t, delta_r]
        
        Returns:
            updated_pose: [B, 4, 4]
        """
        delta_t = delta[:, :3]  # [B, 3]
        delta_r = delta[:, 3:]  # [B, 3]
        
        # 计算增量旋转矩阵
        delta_R = self.rodrigues_rotation(delta_r)  # [B, 3, 3]
        
        # 提取当前pose的R和t
        R = pose[:, :3, :3]
        t = pose[:, :3, 3]
        
        # 左乘更新（在world坐标系下）
        R_new = torch.bmm(delta_R, R)
        t_new = torch.bmm(delta_R, t.unsqueeze(-1)).squeeze(-1) + delta_t
        
        # 构造新pose
        pose_new = pose.clone()
        pose_new[:, :3, :3] = R_new
        pose_new[:, :3, 3] = t_new
        
        return pose_new
    
    def refine_step(self, pose, points_3d, features, intrinsics, image_size, level):
        """
        单步refinement (可微)
        
        使用梯度上升来最大化对齐分数
        """
        # 1. 计算当前对齐分数
        pose.requires_grad_(True)
        score, valid_ratio = self.compute_alignment_score(
            pose, points_3d, features, intrinsics, image_size
        )
        
        # 2. 计算梯度
        loss = -score.mean()  # 负号：梯度上升
        
        # 清空之前的梯度
        if pose.grad is not None:
            pose.grad.zero_()
        
        loss.backward(retain_graph=True)
        
        if pose.grad is None:
            return torch.zeros(pose.shape[0], 6, device=pose.device), score.detach(), valid_ratio.detach()
        
        grad = pose.grad.clone()
        pose.requires_grad_(False)
        
        # 3. 提取6-DoF梯度
        grad_t = grad[:, :3, 3]  # [B, 3]
        
        # 旋转梯度：取反对称部分
        grad_R = grad[:, :3, :3]
        grad_R_skew = (grad_R - grad_R.transpose(-1, -2)) / 2
        grad_r = torch.stack([
            grad_R_skew[:, 2, 1],
            grad_R_skew[:, 0, 2],
            grad_R_skew[:, 1, 0]
        ], dim=-1)
        
        grad_6dof = torch.cat([grad_t, grad_r], dim=-1)  # [B, 6]
        
        # 4. 计算adaptive damping
        grad_stats = torch.abs(grad_6dof)  # [B, 6]
        damping = self.damping_network(level, grad_stats)  # [B, 6]
        
        # 5. 计算更新步长
        lr = torch.abs(self.lr_scale[level])  # 确保正值
        delta = -lr * grad_6dof / (damping + 1e-6)  # [B, 6]
        
        # 6. Clip更新幅度防止发散
        delta = torch.clamp(delta, -0.5, 0.5)
        
        return delta, score.detach(), valid_ratio.detach()
    
    def forward(self, pose, features, points_3d, intrinsics, image_size, 
                level, prob_distribution=None, training=True):
        """
        端到端可微的refinement
        
        Args:
            pose: [B, 4, 4] 初始pose
            features: [B, C, H, W] 特征图
            points_3d: [B, N, 3] 3D点（世界坐标系）
            intrinsics: [B, 3, 3/4] 相机内参
            image_size: (H, W)
            level: stage索引
            prob_distribution: [B, N_samples] 投票概率分布（可选）
            training: 是否训练模式
        
        Returns:
            refined_pose: [B, 4, 4] 优化后的pose
            aux_outputs: dict 包含辅助信息
        """
        B = pose.shape[0]
        
        # 1. 提取refinement专用特征
        refine_features = self.feature_head(features)  # [B, 1, H, W]
        
        # 2. 预测置信度
        if prob_distribution is not None:
            confidence = self.confidence_predictor(
                features, pose, prob_distribution
            )  # [B, 1]
        else:
            # 如果没有prob_distribution，使用默认值
            confidence = torch.ones(B, 1, device=pose.device) * 0.5
        
        # 3. 迭代refinement
        current_pose = pose.detach().clone()  # 断开梯度，fresh start
        
        iteration_info = {
            'deltas': [],
            'scores': [],
            'valid_ratios': []
        }
        
        for it in range(self.max_iterations):
            # 执行一步refinement
            delta, score, valid_ratio = self.refine_step(
                current_pose, points_3d, refine_features, 
                intrinsics, image_size, level
            )
            
            # 应用更新
            current_pose = self.apply_pose_delta(current_pose, delta)
            
            # 记录信息
            iteration_info['deltas'].append(delta)
            iteration_info['scores'].append(score)
            iteration_info['valid_ratios'].append(valid_ratio)
            
            # 早停（仅推理时）
            if not training:
                delta_norm = torch.norm(delta, dim=-1).mean()
                if delta_norm < 1e-4:
                    break
        
        # 4. 使用confidence进行加权blend
        # 高置信度时更多使用refined pose，低置信度时保留原始pose
        confidence_weight = confidence.view(-1, 1, 1)
        blended_pose = confidence_weight * current_pose + (1 - confidence_weight) * pose
        
        # 5. 准备输出
        aux_outputs = {
            'confidence': confidence,  # [B, 1]
            'iteration_info': iteration_info,
            'refined_pose_raw': current_pose,  # blend之前的
            'refine_features': refine_features,
            'final_score': iteration_info['scores'][-1] if iteration_info['scores'] else None,
            'final_valid_ratio': iteration_info['valid_ratios'][-1] if iteration_info['valid_ratios'] else None
        }
        
        return blended_pose, aux_outputs


def test_learnable_refinement():
    """
    测试代码
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模块
    refiner = DifferentiablePoseRefinement(
        num_levels=3,
        max_iterations=5,
        feature_dim=8
    ).to(device)
    
    # 测试数据
    B = 2
    pose = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
    pose[:, :3, 3] = torch.randn(B, 3).to(device) * 10  # 随机平移
    
    features = torch.randn(B, 8, 64, 64).to(device)
    points_3d = torch.randn(B, 100, 3).to(device) * 50
    intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)
    intrinsics[:, 0, 0] = 500  # fx
    intrinsics[:, 1, 1] = 500  # fy
    intrinsics[:, 0, 2] = 320  # cx
    intrinsics[:, 1, 2] = 240  # cy
    
    prob_dist = torch.randn(B, 100).softmax(dim=-1).to(device)
    
    # Forward
    refined_pose, aux = refiner(
        pose, features, points_3d, intrinsics, (64, 64),
        level=0, prob_distribution=prob_dist, training=True
    )
    
    print(f"Input pose shape: {pose.shape}")
    print(f"Refined pose shape: {refined_pose.shape}")
    print(f"Confidence: {aux['confidence']}")
    print(f"Final score: {aux['final_score']}")
    print(f"Test passed!")


if __name__ == '__main__':
    test_learnable_refinement()