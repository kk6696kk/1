import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time
import matplotlib.pyplot as plt
import numpy as np
eps = 1e-12
from PIL import Image
import torchvision.transforms as transforms

class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),nn.BatchNorm2d(outc),nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x,p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0,base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number >  0:
            mod_p_n_x,mod_p_n_y = torch.meshgrid(
                torch.arange(row_number,row_number+1),
                torch.arange(0,mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x,p_n_y  = torch.cat((p_n_x,mod_p_n_x)),torch.cat((p_n_y,mod_p_n_y))
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + self.p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
        
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset
 

# -----------------------------------------------------------------------------
# Building blocks that use LDConv
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
# FeatExtNet with integrated LDConv layers
# -----------------------------------------------------------------------------
class FeatExtNet_LDIntegrated(nn.Module):
    def __init__(self, base_channels=16, num_stage=3):
        super(FeatExtNet_LDIntegrated, self).__init__()
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            LDConvUnit(3, base_channels, num_param=3),
            LDConvUnit(base_channels, base_channels, num_param=3)
        )
        self.conv1 = nn.Sequential(
            LDConvUnit(base_channels, base_channels * 2, num_param=5, stride=2),
            LDConvUnit(base_channels * 2, base_channels * 2, num_param=5)
        )
        self.conv2 = nn.Sequential(
            LDConvUnit(base_channels * 2, base_channels * 4, num_param=7, stride=2),
            LDConvUnit(base_channels * 4, base_channels * 4, num_param=7)
        )

        self.out1 = nn.Conv2d(base_channels * 4, 1, 1, bias=False)
        self.confidence1 = AdaptationBlock(base_channels * 4, 1)

        if num_stage == 3:
            self.deconv1 = Deconv2dBlock(base_channels * 4, base_channels * 2, 3, num_param=5)
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
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        outputs = {}
        out = self.out1(conv2)
        conf = self.confidence1(conv2).sigmoid()
        outputs["stage1_f"] = out
        outputs["stage1_c"] = conf

        if self.num_stage == 3:
            intra_feat = self.deconv1(conv1, conv2)
            out = self.out2(intra_feat)
            conf = self.confidence2(intra_feat).sigmoid()
            outputs["stage2_f"] = out
            outputs["stage2_c"] = conf

            intra_feat = self.deconv2(conv0, intra_feat)
            out = self.out3(intra_feat)
            conf = self.confidence3(intra_feat).sigmoid()
            outputs["stage3_f"] = out
            outputs["stage3_c"] = conf

            inp_fine = torch.cat((intra_feat, out, x), dim=1)
            out_fine = self.fine_conv(inp_fine)
            outputs["stage_fine"] = out_fine

        return outputs


#if __name__ == '__main__':
#    device = torch.device("cuda:0")
#    model = FeatExtNet_LDIntegrated(base_channels=16, num_stage=3)
#    x = torch.randn(1, 3, 256, 256)
#    model = model.to(device)
#    x = x.to(device)
#    y = model(x)
#    for k, v in y.items():
#        print(k, v.shape)

