import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange
import math

eps = 1e-12

class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)


class Conv2dUnit(nn.Module):
    """标准卷积单元"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv2dUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ImprovedLDConv(nn.Module):
    """改进的可变形卷积 - 修复初始化和感受野问题"""
    def __init__(self, inc, outc, kernel_size=3, stride=1, dilation=1, groups=1):
        super(ImprovedLDConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        # 使用标准卷积核大小
        self.weight = nn.Parameter(torch.Tensor(outc, inc // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(outc))
        
        # 偏移量预测 - 2 * kernel_size^2 个通道 (x和y方向)
        offset_channels = 2 * kernel_size * kernel_size * groups
        self.offset_conv = nn.Conv2d(
            inc, 
            offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=True
        )
        
        # 改进的初始化策略
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.bn = nn.BatchNorm2d(outc)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        # 使用PyTorch的deform_conv2d (需要torchvision)
        try:
            from torchvision.ops import deform_conv2d
            out = deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.stride,
                padding=self.kernel_size//2,
                dilation=self.dilation
            )
        except ImportError:
            # 降级到标准卷积
            print("Warning: torchvision not available, using standard conv")
            out = F.conv2d(x, self.weight, self.bias, 
                          stride=self.stride, 
                          padding=self.kernel_size//2)
        
        out = self.bn(out)
        out = F.silu(out)
        return out


class EdgeAwareConv(nn.Module):
    """边缘感知卷积 - 针对线框特征优化"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EdgeAwareConv, self).__init__()
        self.stride = stride
        
        # 主分支 - 标准卷积
        self.main_conv = Conv2dUnit(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size//2
        )
        
        # 边缘检测分支
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        main_feat = self.main_conv(x)
        edge_weight = self.edge_conv(x)
        
        if self.stride > 1:
            edge_weight = F.avg_pool2d(edge_weight, self.stride)
        
        return main_feat * (1 + edge_weight)


class MultiScaleConv(nn.Module):
    """多尺度卷积 - 捕获不同尺度的线框特征"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(MultiScaleConv, self).__init__()
        mid_channels = out_channels // 4
        
        self.branch1 = Conv2dUnit(in_channels, mid_channels, 1, stride=stride)
        self.branch2 = Conv2dUnit(in_channels, mid_channels, 3, stride=stride, padding=1)
        self.branch3 = Conv2dUnit(in_channels, mid_channels, 3, stride=stride, padding=2, dilation=2)
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1),
            Conv2dUnit(in_channels, mid_channels, 1)
        )
        
        self.fusion = Conv2dUnit(out_channels, out_channels, 1)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        return out


class Deconv2dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                       stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ImprovedDeconv2dBlock(nn.Module):
    """改进的上采样块 - 使用多尺度融合"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ImprovedDeconv2dBlock, self).__init__()
        
        self.deconv = Deconv2dUnit(
            in_channels, out_channels, kernel_size, 
            stride=2, padding=1, output_padding=1
        )
        
        # 使用多尺度卷积进行特征融合
        self.fusion = MultiScaleConv(2 * out_channels, out_channels)
        
    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.fusion(x)
        return x


class FeatExtNet(nn.Module):
    """改进的特征提取网络"""
    def __init__(self, base_channels=16, num_stage=3, use_deformable=False):
        super(FeatExtNet, self).__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage
        self.use_deformable = use_deformable

        # 编码器 - 使用边缘感知卷积
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            EdgeAwareConv(base_channels, base_channels, 3, 1),
        )
        
        # 第一层下采样 - 可选择使用可变形卷积
        if use_deformable:
            self.conv1 = nn.Sequential(
                ImprovedLDConv(base_channels, base_channels * 2, 3, stride=2),
                EdgeAwareConv(base_channels * 2, base_channels * 2, 3),
                EdgeAwareConv(base_channels * 2, base_channels * 2, 3)
            )
        else:
            self.conv1 = nn.Sequential(
                MultiScaleConv(base_channels, base_channels * 2, stride=2),
                EdgeAwareConv(base_channels * 2, base_channels * 2, 3),
                EdgeAwareConv(base_channels * 2, base_channels * 2, 3)
            )
        
        # 第二层下采样
        self.conv2 = nn.Sequential(
            MultiScaleConv(base_channels * 2, base_channels * 4, stride=2),
            EdgeAwareConv(base_channels * 4, base_channels * 4, 3),
            EdgeAwareConv(base_channels * 4, base_channels * 4, 3)
        )

        self.out1 = nn.Conv2d(base_channels * 4, 1, 1, bias=False)
        self.out_channels = [base_channels]
        self.confidence1 = AdaptationBlock(base_channels * 4, 1)

        if num_stage == 3:
            self.deconv1 = ImprovedDeconv2dBlock(base_channels * 4, base_channels * 2, 3)
            self.deconv2 = ImprovedDeconv2dBlock(base_channels * 2, base_channels, 3)

            self.out2 = nn.Conv2d(base_channels * 2, 1, 1, bias=False)
            self.out3 = nn.Conv2d(base_channels, 1, 1, bias=False)
            
            # 改进的精细化卷积 - 使用多尺度融合
            self.fine_conv = nn.Sequential(
                MultiScaleConv(base_channels + 4, (base_channels + 4) * 2),
                EdgeAwareConv((base_channels + 4) * 2, base_channels + 4, 3),
                nn.Conv2d(base_channels + 4, 1, 1)
            )
            
            self.confidence2 = AdaptationBlock(base_channels * 2, 1)
            self.confidence3 = AdaptationBlock(base_channels, 1)
            self.out_channels.append(base_channels)
            self.out_channels.append(base_channels)

        elif num_stage == 2:
            self.deconv1 = ImprovedDeconv2dBlock(base_channels * 4, base_channels * 2, 3)
            self.out2 = nn.Conv2d(base_channels * 2, 1, 1, bias=False)
            self.confidence2 = AdaptationBlock(base_channels * 2, 1)
            self.out_channels.append(base_channels)
        
        self.is_optimized_forward = True
    
    def toggle_optimization(self, enabled=True):
        self.is_optimized_forward = enabled
        return self
    
    def _enhanced_global_context(self, x):
        """增强的全局上下文 - 添加通道注意力"""
        # 全局平均池化
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        
        # 通道注意力
        scale = torch.sigmoid(avg_pool + max_pool)
        return x * scale

    def forward(self, x):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # 编码器
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        # 全局上下文增强
        intra_feat = conv2
        if self.is_optimized_forward:
            intra_feat = self._enhanced_global_context(intra_feat)
        
        outputs = {}
        out = self.out1(intra_feat)
        conf = self.confidence1(intra_feat).sigmoid()

        outputs["stage1_f"] = out
        outputs["stage1_c"] = conf
        
        if self.num_stage == 3:
            intra_feat = self.deconv1(conv1, intra_feat)
            if self.is_optimized_forward:
                intra_feat = self._enhanced_global_context(intra_feat)
                
            out = self.out2(intra_feat)
            conf = self.confidence2(intra_feat).sigmoid()
            outputs["stage2_f"] = out
            outputs["stage2_c"] = conf

            intra_feat = self.deconv2(conv0, intra_feat)
            if self.is_optimized_forward:
                intra_feat = self._enhanced_global_context(intra_feat)
                
            out = self.out3(intra_feat)
            conf = self.confidence3(intra_feat).sigmoid()
            outputs["stage3_f"] = out
            outputs["stage3_c"] = conf
            
            # 特征融合
            inp_fine = torch.cat((intra_feat, out, x), dim=1)
            out_fine = self.fine_conv(inp_fine)
            
            # 残差连接
            if self.is_optimized_forward and out.shape == out_fine.shape:
                out_fine = out_fine + out * 0.2
                
            outputs["stage_fine"] = out_fine

        elif self.num_stage == 2:
            intra_feat = self.deconv1(conv1, intra_feat)
            if self.is_optimized_forward:
                intra_feat = self._enhanced_global_context(intra_feat)
                
            out = self.out2(intra_feat)
            conf = self.confidence2(intra_feat).sigmoid()
            outputs["stage2_f"] = out
            outputs["stage2_c"] = conf
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        if hasattr(self, 'debug_timing') and self.debug_timing:
            print(f"FeatExtNet forward pass time: {(end_time - start_time) * 1000:.2f}ms")
        
        return outputs


