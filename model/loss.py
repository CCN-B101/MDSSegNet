import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

class DiceCELoss(nn.Module):
    """
    Dice + CrossEntropy Loss with optional label smoothing
    """
    def __init__(self, ignore_index=255, weight=None, smooth=1e-5, label_smoothing=0.1):
        super(DiceCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.smooth = smooth
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W], targets: [B, H, W]
        num_classes = inputs.shape[1]
        inputs_soft = F.softmax(inputs, dim=1)

        # CrossEntropy Loss with label smoothing
        ce_loss = F.cross_entropy(
            inputs, targets,
            ignore_index=self.ignore_index,
            weight=self.weight,
            label_smoothing=self.label_smoothing
        )

        # Dice Loss
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        cardinality = torch.sum(inputs_soft + targets_one_hot, dims)
        dice_loss = 1.0 - ((2. * intersection + self.smooth) / (cardinality + self.smooth)).mean()

        return ce_loss + dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 将目标转换为one-hot编码
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 计算Focal Loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# class BoundaryLoss(nn.Module):
#     def __init__(self):
#         super(BoundaryLoss, self).__init__()
#
#     def forward(self, inputs, targets):
#         # 将目标转换为one-hot编码
#         num_classes = inputs.shape[1]
#         targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
#
#         # 对每个类别单独计算边界损失
#         inputs_soft = F.softmax(inputs, dim=1)
#
#         # 只对前景类别（裂缝）计算边界损失
#         boundary_loss = 0
#         for class_idx in range(1, num_classes):  # 跳过背景类别（0）
#             pred_class = inputs_soft[:, class_idx:class_idx + 1, :, :]
#             target_class = targets_one_hot[:, class_idx:class_idx + 1, :, :]
#
#             # 将目标转换为距离变换图
#             target_np = target_class.cpu().numpy()
#             dist_maps = torch.zeros_like(target_class)
#
#             for i in range(target_class.shape[0]):
#                 # 计算前景的距离变换
#                 pos_dist = distance_transform_edt(1 - target_np[i, 0, :, :])
#                 # 计算背景的距离变换
#                 neg_dist = distance_transform_edt(target_np[i, 0, :, :])
#
#                 # 合并得到边界距离图
#                 dist_map = torch.from_numpy(pos_dist + neg_dist).to(inputs.device)
#                 dist_maps[i, 0, :, :] = dist_map
#
#             # 计算边界损失
#             loss = pred_class * dist_maps
#             boundary_loss += loss.mean()
#
#         return boundary_loss / (num_classes - 1)  # 平均所有前景类别的损失


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.cldice_loss = DiceCELoss()
        # self.boundary_loss = BoundaryLoss()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # 确保输入和目标形状匹配
        assert inputs.shape[0] == targets.shape[0], "Batch size mismatch"
        assert inputs.shape[2] == targets.shape[1], "Height mismatch"
        assert inputs.shape[3] == targets.shape[2], "Width mismatch"
        assert inputs.shape[1] == self.num_classes, f"Expected {self.num_classes} classes, got {inputs.shape[1]}"

        focal = self.focal_loss(inputs, targets)
        cldice = self.cldice_loss(inputs, targets)
        # boundary = self.boundary_loss(inputs, targets)

        total_loss = (focal * 0.4 + cldice * 0.6)

        return total_loss