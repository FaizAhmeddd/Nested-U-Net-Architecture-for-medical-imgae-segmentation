import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from lovasz_losses import lovasz_hinge


__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'IoULoss', 'DiceLoss', 'CombinedLoss']


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class CombinedLoss(nn.Module):
    def __init__(self, weight_bce_dice=0.5, weight_lovasz=0.5, weight_iou=0.5, weight_dice=0.5):
        super().__init__()
        self.weight_bce_dice = weight_bce_dice
        self.weight_lovasz = weight_lovasz
        self.weight_iou = weight_iou
        self.weight_dice = weight_dice
        self.bce_dice_loss = BCEDiceLoss()
        self.lovasz_loss = LovaszHingeLoss()
        self.iou_loss = IoULoss()
        self.dice_loss = DiceLoss()

    def forward(self, input, target):
        bce_dice_loss = self.bce_dice_loss(input, target)
        lovasz_loss = self.lovasz_loss(input, target)
        iou_loss = self.iou_loss(input, target)
        dice_loss = self.dice_loss(input, target)
        loss = (
            self.weight_bce_dice * bce_dice_loss +
            self.weight_lovasz * lovasz_loss +
            self.weight_iou * iou_loss +
            self.weight_dice * dice_loss
        )
        return loss
