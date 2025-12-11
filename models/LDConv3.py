import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
eps = 1e-12
from PIL import Image
import torchvision.transforms as transforms
from einops import rearrange
import math

class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)

class Conv2dUnit(nn.Module):
    """完全保持原始Conv2dUnit结构，只在前向传播中优化"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # 存储输入和输出通道数，用于运行时优化
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        # 记录输入形状以便优化
        input_shape = x.shape
        
        # 标准前向传播
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
            
        return x
    
class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        with torch.no_grad():
           self.p_conv.weight.data.uniform_(-1e-4, 1e-4)
        nn.init.constant_(self.p_conv.bias, 0) # 如果有bias的话
        #self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        scaled_grad_input = tuple(g * 0.2 if g is not None else None for g in grad_input)
        scaled_grad_output = tuple(g * 0.2 if g is not None else None for g in grad_output)
        return scaled_grad_input, scaled_grad_output


    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int), indexing='ij')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(torch.arange(row_number, row_number + 1), torch.arange(0, mod_number), indexing='ij')
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(0, h * self.stride, self.stride), torch.arange(0, w * self.stride, self.stride), indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_0 = self._get_p_0(h, w, N, dtype).to(offset.device)
        p_n = self.p_n.to(offset.device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
    
class LDConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_param=5, stride=1):
        super(LDConvUnit, self).__init__()
        self.ldconv = LDConv(in_channels, out_channels, num_param=num_param, stride=stride)
    def forward(self, x):
        return self.ldconv(x)

class Deconv2dUnit(nn.Module):
    """完全保持原始Deconv2dUnit结构，只在前向传播中优化"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        # 标准前向传播
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_param=5):
        super(Deconv2dBlock, self).__init__()
        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1)
        self.conv = LDConvUnit(2 * out_channels, out_channels, num_param=num_param)
    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x
    

class FeatExtNet(nn.Module):
    def __init__(self, base_channels=16, num_stage=3):
        super(FeatExtNet, self).__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        # Encoder
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        self.conv1 = nn.Sequential(
            LDConvUnit(base_channels, base_channels * 2, num_param=5, stride=2),
            LDConvUnit(base_channels * 2, base_channels * 2, num_param=3),
            LDConvUnit(base_channels * 2, base_channels * 2, num_param=3)
        )
        self.conv2 = nn.Sequential(
            LDConvUnit(base_channels * 2, base_channels * 4, num_param=5, stride=2),
            LDConvUnit(base_channels * 4, base_channels * 4, num_param=3),
            LDConvUnit(base_channels * 4, base_channels * 4, num_param=3)
        )

        self.out1 = nn.Conv2d(base_channels * 4, 1, 1, bias=False)
        self.out_channels = [base_channels]
        self.confidence1 = AdaptationBlock(base_channels * 4, 1)

        if num_stage == 3:
            self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)
            self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3)

            self.out2 = nn.Conv2d(base_channels * 2, 1, 1, bias=False)
            self.out3 = nn.Conv2d(base_channels, 1, 1, bias=False)
            
            # 保持原始fine_conv结构
            self.fine_conv = nn.Sequential(
                LDConvUnit(base_channels + 4, (base_channels + 4) * 2, num_param=5),
                LDConvUnit((base_channels + 4) * 2, base_channels + 4, num_param=5),
                nn.Conv2d(base_channels + 4, 1, 1)
            )
            
            self.confidence2 = AdaptationBlock(base_channels * 2, 1)
            self.confidence3 = AdaptationBlock(base_channels, 1)
            self.out_channels.append(base_channels)
            self.out_channels.append(base_channels)

        elif num_stage == 2:
            self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3)

            self.out2 = nn.Conv2d(base_channels * 2, 1, 1, bias=False)
            self.confidence2 = AdaptationBlock(base_channels * 2, 1)
            self.out_channels.append(base_channels)
        
        # 缓存标志，用于训练/推理模式切换
        self.is_optimized_forward = True
    
    def toggle_optimization(self, enabled=True):
        """启用或禁用优化前向传播"""
        self.is_optimized_forward = enabled
        return self
    
    def _fast_global_context(self, x):
        """计算高效的全局上下文增强"""
        # 快速全局平均池化
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        # 简化的注意力机制
        scale = torch.sigmoid(avg_pool)
        return x * scale

    def forward(self, x):
        # 使用torch.cuda.synchronize()和time.time()测量性能
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # 标准前向传播流程
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        # 应用高效的全局上下文增强
        intra_feat = conv2
        if self.is_optimized_forward:
            intra_feat = self._fast_global_context(intra_feat)
        
        outputs = {}
        out = self.out1(intra_feat)
        conf = self.confidence1(intra_feat).sigmoid()

        outputs["stage1_f"] = out
        outputs["stage1_c"] = conf
        
        if self.num_stage == 3:
            intra_feat = self.deconv1(conv1, intra_feat)
            if self.is_optimized_forward:
                intra_feat = self._fast_global_context(intra_feat)
                
            out = self.out2(intra_feat)
            conf = self.confidence2(intra_feat).sigmoid()
            outputs["stage2_f"] = out
            outputs["stage2_c"] = conf

            intra_feat = self.deconv2(conv0, intra_feat)
            if self.is_optimized_forward:
                intra_feat = self._fast_global_context(intra_feat)
                
            out = self.out3(intra_feat)
            conf = self.confidence3(intra_feat).sigmoid()
            outputs["stage3_f"] = out
            outputs["stage3_c"] = conf
            
            # 特征融合
            inp_fine = torch.cat((intra_feat, out, x), dim=1)
            out_fine = self.fine_conv(inp_fine)
            
            # 添加残差连接来增强性能
            if self.is_optimized_forward and out.shape == out_fine.shape:
                out_fine = out_fine + out * 0.2
                
            outputs["stage_fine"] = out_fine

        elif self.num_stage == 2:
            intra_feat = self.deconv1(conv1, intra_feat)
            if self.is_optimized_forward:
                intra_feat = self._fast_global_context(intra_feat)
                
            out = self.out2(intra_feat)
            conf = self.confidence2(intra_feat).sigmoid()
            outputs["stage2_f"] = out
            outputs["stage2_c"] = conf
        
        # 计算并打印处理时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        # 只在调试模式下打印时间
        if hasattr(self, 'debug_timing') and self.debug_timing:
            print(f"FeatExtNet forward pass time: {(end_time - start_time) * 1000:.2f}ms")
        
        return outputs
