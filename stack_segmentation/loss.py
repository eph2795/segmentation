import torch
from torch import nn


def dice_coef_loss(y_pred, y_true, smooth=1.0):
    intersection = (y_true * y_pred[:, 1]).sum()
    return 1 - (2. * intersection + smooth) / (y_true.sum() + y_pred[:, 1].sum() + smooth)


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
            
    def loss_func(y_pred, y_true):
        return sum([coeff * loss(y_pred, y_true) for loss, coeff in zip(funcs, coeffs)])
    
    return loss_func