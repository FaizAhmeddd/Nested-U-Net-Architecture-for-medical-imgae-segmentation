import torch


def iou_score(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output)
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum().float()
    union = (output_ | target_).sum().float()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1)
    target = target.view(-1)
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def accuracy(output, target):
    output = torch.sigmoid(output)
    output_ = output > 0.5
    target_ = target > 0.5

    correct = (output_ == target_).sum().item()
    total = output.numel()

    return correct / total


def precision(output, target):
    output = torch.sigmoid(output)
    output_ = output > 0.5
    target_ = target > 0.5

    true_positive = (output_ & target_).sum().item()
    predicted_positive = output_.sum().item()

    return true_positive / (predicted_positive + 1e-5)


def recall(output, target):
    output = torch.sigmoid(output)
    output_ = output > 0.5
    target_ = target > 0.5

    true_positive = (output_ & target_).sum().item()
    actual_positive = target_.sum().item()

    return true_positive / (actual_positive + 1e-5)


def f1_score(output, target):
    prec = precision(output, target)
    rec = recall(output, target)

    return 2.0 * (prec * rec) / (prec + rec + 1e-5)
