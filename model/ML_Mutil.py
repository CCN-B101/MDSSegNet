import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.net import mobilenetv2


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]   # 去掉最后的conv+pool等

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]  # 仅供参考（官方结构分界）

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        """
        顺序前向，返回四个尺度（约 1/4、1/8、1/16、1/32 或膨胀后的等效感受野）：
        l1:  ~1/4,  通道约 24
        l2:  ~1/8,  通道约 32
        l3:  ~1/16, 通道约 64
        l4:  ~1/32(或dilated 1/16), 通道约 320
        """
        l1 = self.features[:4](x)        # ~1/4
        l2 = self.features[4:7](l1)      # ~1/8
        l3 = self.features[7:11](l2)     # ~1/16
        l4 = self.features[11:](l3)      # ~1/32(或等效膨胀)
        return l1, l2, l3, l4


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, g=1, use_coord=False):
        super().__init__()
        self.use_coord = use_coord
        self.conv = nn.Conv2d(in_c + (2 if use_coord else 0), out_c, k, s, p, bias=False, groups=g)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_coord:
            b, _, h, w = x.size()
            xx_channel = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
            yy_channel = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
            x = torch.cat([x, xx_channel, yy_channel], dim=1)
        return self.relu(self.bn(self.conv(x)))


class SeBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        m = max(c // r, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, m, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(m, c, 1, bias=True), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


# -------------------------------------------------------------
# CoordAtt
# -------------------------------------------------------------
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # (n,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # (n,c,1,w)

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, 1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, 1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                  # (n,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (n,c,w,1)
        y = torch.cat([x_h, x_w], dim=2)      # (n,c,h+w,1)
        y = self.act(self.bn1(self.conv1(y))) # (n,mip,h+w,1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)         # (n,mip,1,w)
        a_h = self.conv_h(x_h).sigmoid()      # (n,oup,h,1)
        a_w = self.conv_w(x_w).sigmoid()      # (n,oup,1,w)
        return x * a_h * a_w


# -------------------------------------------------------------
# Detail Branch
# -------------------------------------------------------------
class DetailBranch(nn.Module):
    def __init__(self, c1=64, c2=128, c3=128, c4=128, c5=128):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBNReLU(3, c1, 3, 2, 1),          # -> H/2
            ConvBNReLU(c1, c2, 3, 1, 1),
        )
        self.s2 = nn.Sequential(
            ConvBNReLU(c2, c2, 3, 2, 1, use_coord=False),  # -> H/4
            ConvBNReLU(c2, c2, 3, 1, 1, use_coord=True),
        )
        self.s3 = nn.Sequential(
            ConvBNReLU(c2, c3, 3, 2, 1, use_coord=False),  # -> H/8
            ConvBNReLU(c3, c3, 3, 1, 1, use_coord=True),
        )
        self.s4 = nn.Sequential(
            ConvBNReLU(c3, c4, 3, 2, 1, use_coord=False),  # -> H/16
            ConvBNReLU(c4, c4, 3, 1, 1, use_coord=True),
        )
        # 主路径
        self.s5_main = nn.Sequential(
            ConvBNReLU(c4, c5, 3, 2, 1, use_coord=True),   # -> H/32
            ConvBNReLU(c5, c5, 3, 1, 1, use_coord=True),
        )
        # 空间增强路径
        self.s5_spatial = nn.Sequential(
            ConvBNReLU(c4, c5, 3, 2, 1, use_coord=True),   # -> H/32
            CoordAtt(c5, c5),
        )
        # 融合权重（自适应）
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始语义/空间各一半

    def forward(self, x):
        f1 = self.s1(x)          # (B,128,H/2,W/2)
        f2 = self.s2(f1)         # (B,128,H/4,W/4)   -> fs1
        f3 = self.s3(f2)         # (B,128,H/8,W/8)   -> fs2
        f4 = self.s4(f3)         # (B,128,H/16,W/16) -> fs3
        f5_main = self.s5_main(f4)      # (B,128,H/32,W/32) -> fs4 (上采到1/16用于s5)
        f5_spatial = self.s5_spatial(f4)# (B,128,H/32,W/32)

        # 自适应融合 (sigmoid 归一化)
        w = torch.sigmoid(self.alpha)
        f5 = torch.cat([(1 - w) * f5_main, w * f5_spatial], dim=1)  # (B,256,H/32,W/32) -> fs5

        # 返回顺序：与下游一致
        return f1, f2, f3, f4, f5_main, f5    # f2..f5_main..f5 -> fs1..fs5


# -------------------------------------------------------------
# Context Branch (MobileNetV2 + ARMs + 1x1 Reduce)  —— No Internal Fusion
# -------------------------------------------------------------
class AttentionRefine(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.sig(self.bn(self.conv(w)))
        return x * w


class ContextBranch(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = MobileNetV2(pretrained=pretrained)

        # ARMs
        self.arm_low  = AttentionRefine(24, 24)    # ~1/4
        self.arm_mid  = AttentionRefine(32, 32)    # ~1/8
        self.arm_high = AttentionRefine(64, 64)    # ~1/16
        self.arm_top  = AttentionRefine(320, 320)  # ~1/32

        # Reduce to 128c
        self.reduce_low  = ConvBNReLU(24, 128, 1, 1, 0)
        self.reduce_mid  = ConvBNReLU(32, 128, 1, 1, 0)
        self.reduce_high = ConvBNReLU(64, 128, 1, 1, 0)
        self.reduce_top  = ConvBNReLU(320, 128, 1, 1, 0)

    def forward(self, x):
        # Backbone 多尺度
        l1, l2, l3, l4 = self.backbone(x)   # ~1/4, ~1/8, ~1/16, ~1/32(或等效)

        # ARM
        low_feat  = self.arm_low(l1)
        mid_feat  = self.arm_mid(l2)
        high_feat = self.arm_high(l3)
        top_feat  = self.arm_top(l4)

        # 统一到 128 通道（直接作为四个输出返回）
        low_red  = self.reduce_low(low_feat)    # -> ls1 : 1/4,  128c
        mid_red  = self.reduce_mid(mid_feat)    # -> ls2 : 1/8,  128c
        high_red = self.reduce_high(high_feat)  # -> ls3 : 1/16, 128c
        top_red  = self.reduce_top(top_feat)    # -> ls4 : 1/32(或等效1/16), 128c

        # 不做 1/16 内部融合，直接返回 4 个尺度
        return low_red, mid_red, high_red, top_red   # ls1, ls2, ls3, ls4


# -------------------------------------------------------------
# UNet-style Feature Fusion Block
# -------------------------------------------------------------
class FusionBlock(nn.Module):
    def __init__(self, in_detail, in_context, out):
        super().__init__()
        self.proj = ConvBNReLU(in_detail + in_context, out, 1, 1, 0)
        self.se = SeBlock(out)
        self.refine = ConvBNReLU(out, out, 3, 1, 1)

    def forward(self, d, c, prev_fused=None):
        if c.shape[-2:] != d.shape[-2:]:
            c = F.interpolate(c, size=d.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([d, c], dim=1)  # 通道 = in_detail + in_context
        if prev_fused is not None:
            # 这里要求 prev_fused 通道数 == out（或在投影前维度一致）
            x = x + F.interpolate(prev_fused, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.proj(x)   # -> out
        x = self.se(x)
        x = self.refine(x) # -> out
        return x


# -------------------------------------------------------------
# Segmentation Head
# -------------------------------------------------------------
class SegHead(nn.Module):
    def __init__(self, in_c, mid_c, n_classes):
        super().__init__()
        self.conv = w�h��춻�q�^wHeads
        self.head_s5 = SegHead(256, 128, n_classes)
        self.head_s4 = SegHead(256, 128, n_classes)
        self.head_s3 = SegHead(256, 128, n_classes)
        self.head_s2 = SegHead(256, 128, n_classes)

    def forward(self, x):
        size = x.shape[-2:]
        # Detail 分支
        f1, f2, f3, f4, f5_main, f5 = self.detail(x)  # fs1=f2, fs2=f3, fs3=f4, fs4=f5_main, fs5=f5

        # Context 分支（无内融合）
        ls1, ls2, ls3, ls4 = self.context(x)  # 1/4, 1/8, 1/16, 1/32(或等效)

        # ---- UNet-style 同尺度融合 ----
        # s5 (~1/16)：fs4(=f5_main, 1/32→1/16) 与 up(ls4→1/16) 融合，prev=fs5(=f5,256c)
        c_for_s5 = F.interpolate(ls4, size=f5_main.shape[-2:], mode='bilinear', align_corners=False)
        s5 = self.fuse_s5(f5_main, c_for_s5, prev_fused=f5)   # -> 256c, ~1/16

        # s4 (~1/16)：fs3 与 ls3 融合
        s4 = self.fuse_s4(f4, ls3, prev_fused=s5)             # -> 256c, ~1/16

        # s3 (~1/8)：fs2 与 ls2 融合
        s3 = self.fuse_s3(f3, ls2, prev_fused=s4)             # -> 256c, ~1/8

        # s2 (~1/4)：fs1 与 ls1 融合
        s2 = self.fuse_s2(f2, ls1, prev_fused=s3)             # -> 256c, ~1/4

        # 预测头：全部上采到原图
        out_s5 = self.head_s5(s5, size=size)
        out_s4 = self.head_s4(s4, size=size)
        out_s3 = self.head_s3(s3, size=size)
        out_s2 = self.head_s2(s2, size=size)

        if self.aux_mode == 'train':
            # 约定 out_s2 为主分支，其余为辅助
            return out_s2, out_s3, out_s4, out_s5
        else:
            return out_s2


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 3, 448, 448).to(device)
    model = MLiteUNet(n_classes=2, aux_mode='train', pretrained_backbone=False).to(device)
    y = model(x)
    if isinstance(y, tuple):
        print('Outputs:', [t.shape for t in y])
    else:
        print('Output:', y.shape)
    # Count params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable params:', n_params / 1e6, 'M')


#空间分支：fs1=f2(1/4)、fs2=f3(1/8)、fs3=f4(1/16)、fs4=f5_main(1/32→1/16)、fs5=f5(1/32,256c)

#语义分支：ls1=low_red(1/4)、ls2=mid_red(1/8)、ls3=high_red(1/16)、ls4=top_red(1/32)

#融合：s5(fs4 ⟷ up(ls4), prev=fs5) → s4(fs3 ⟷ ls3) → s3(fs2 ⟷ ls2) → s2(fs1 ⟷ ls1)