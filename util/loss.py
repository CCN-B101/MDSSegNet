import torch
import torch.nn as nn
import torch.nn.functional as F


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
