import torch
from torch import nn
import torch.nn.functional as F


def dice_coef_loss(y_pred, y_true, smooth=1.0):
    y_pred = y_pred.softmax(dim=1)[:, 1]
#     y_true = F.one_hot(y_true, num_classes=2).permute(0, 3, 1, 2)
    intersection = (y_true * y_pred).sum()
    return 1 - (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def dice_coef_loss_log(y_pred, y_true, smooth=1.0):
    y_pred = y_pred.softmax(dim=1)[:, 1]
#     y_true = F.one_hot(y_true, num_classes=2).permute(0, 3, 1, 2)
    intersection = (y_true * y_pred).sum()
    return -torch.log((2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth))


def make_loss(loss_list, weight, device):
    
    funcs = []
    coeffs = []
    for loss, coeff in loss_list:
        coeffs.append(coeff)
        if loss == 'BCE':
            if weight is not None:
                weight = torch.FloatTensor(weight).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            funcs.append(criterion)
        elif loss == 'Dice':
            funcs.append(dice_coef_loss)
        elif loss == 'Dice_log':
            funcs.append(dice_coef_loss_log)
            
    def loss_func(y_pred, y_true):
        return sum([coeff * loss(y_pred, y_true) for loss, coeff in zip(funcs, coeffs)])
    
    return loss_func