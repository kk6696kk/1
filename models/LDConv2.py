import math
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------------------------------------------------------
# 新增模块：CBAM 注意力机制
# -----------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# -----------------------------------------------------------------------------
# 新增模块：SPP 全局上下文增强
# -----------------------------------------------------------------------------
class SPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 4]):
        super(SPPModule, self).__init__()
        self.pool_layers = nn.ModuleList()
        for size in pool_sizes:
            self.pool_layers.append(nn.AdaptiveAvgPool2d(size))
        
        # 1x1 卷积用于在拼接后统一通道数
        total_channels = in_channels * (len(pool_sizes) + 1)
        self.conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # 将原始特征和不同尺度的池化特征拼接
        features = [x]
        for pool_layer in self.pool_layers:
            pooled = pool_layer(x)
            # 上采样回原始尺寸以便拼接
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            features.append(upsampled)
        
        concatenated = torch.cat(features, dim=1)
        return self.conv(concatenated)

# -----------------------------------------------------------------------------
# LDConv 核心模块 (这部分无需修改)
# -----------------------------------------------------------------------------
class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

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

# -----------------------------------------------------------------------------
# 辅助构建模块 (这部分无需修改)
# -----------------------------------------------------------------------------
class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)

class LDConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_param=5, stride=1):
        super(LDConvUnit, self).__init__()
        self.ldconv = LDConv(in_channels, out_channels, num_param=num_param, stride=stride)
    def forward(self, x):
        return self.ldconv(x)

class Deconv2dUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu
    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.relu(y, inplace=True)
        return y

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

# -----------------------------------------------------------------------------
# 最终版特征提取网络：集成 SPP 和 CBAM
# -----------------------------------------------------------------------------
class FeatExtNet_LDIntegrated(nn.Module):
    def __init__(self, base_channels=8, num_stage=3):
        super(FeatExtNet_LDIntegrated, self).__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        # Encoder
        self.conv0 = nn.Sequential(
            LDConvUnit(3, base_channels, num_param=3),
            LDConvUnit(base_channels, base_channels, num_param=3)
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

        # === 新增：在网络瓶颈处加入 SPP 和 CBAM ===
        bottleneck_channels = base_channels * 4
        self.spp = SPPModule(bottleneck_channels, bottleneck_channels, pool_sizes=[1, 2, 4])
        self.attention = CBAM(bottleneck_channels)
        # ==========================================
                                                                             
        # Output heads for different stages
        self.out1 = nn.Conv2d(bottleneck_channels, 1, 1, bias=False) # 输入来自增强后的瓶颈特征
        self.confidence1 = AdaptationBlock(bottleneck_channels, 1)

        if num_stage == 3:
            # Decoder
            self.deconv1 = Deconv2dBlock(bottleneck_channels, base_channels * 2, 3, num_param=3)
            self.deconv2 = Deconv2dBlock(base_channels * 2, base_channels, 3, num_param=3)

            self.out2 = nn.Conv2d(base_channels * 2, 1, 1, bias=False)
            self.out3 = nn.Conv2d(base_channels, 1, 1, bias=False)
            
            self.fine_conv = nn.Sequential(
                LDConvUnit(base_channels + 4, (base_channels + 4) * 2, num_param=5),
                LDConvUnit((base_channels + 4) * 2, base_channels + 4, num_param=5),
                nn.Conv2d(base_channels + 4, 1, 1)
            )
            
            self.confidence2 = AdaptationBlock(base_channels * 2, 1)
            self.confidence3 = AdaptationBlock(base_channels, 1)

    def forward(self, x):
        # Encoder path
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        
        # === 新增：应用 SPP 和 CBAM 增强瓶颈特征 ===
        bottleneck_feat = self.spp(conv2)
        bottleneck_feat_enhanced = self.attention(bottleneck_feat)
        # ==========================================
        
        outputs = {}
        
        # Stage 1 output (使用增强后的特征)
        intra_feat_s1 = bottleneck_feat_enhanced
        out_s1 = self.out1(intra_feat_s1)
        conf_s1 = self.confidence1(intra_feat_s1).sigmoid()
        outputs["stage1_f"] = torch.sigmoid(out_s1)
        outputs["stage1_c"] = conf_s1

        if self.num_stage == 3:
            # Decoder path
            # Stage 2 output
            intra_feat_s2 = self.deconv1(conv1, intra_feat_s1) # <--- deconv1的输入现在是增强后的特征
            out_s2 = self.out2(intra_feat_s2)
            conf_s2 = self.confidence2(intra_feat_s2).sigmoid()
            outputs["stage2_f"] = torch.sigmoid(out_s2)
            outputs["stage2_c"] = conf_s2
                            
            # Stage 3 output
            intra_feat_s3 = self.deconv2(conv0, intra_feat_s2)  
            out_s3 = self.out3(intra_feat_s3)
            conf_s3 = self.confidence3(intra_feat_s3).sigmoid()
            outputs["stage3_f"] = torch.sigmoid(out_s3)
            outputs["stage3_c"] = conf_s3
            
            # Final refinement stage
            inp_fine = torch.cat((intra_feat_s3, out_s3, x), dim=1)
            out_fine = self.fine_conv(inp_fine)
            outputs["stage_fine"] = torch.sigmoid(out_fine)

        return outputs