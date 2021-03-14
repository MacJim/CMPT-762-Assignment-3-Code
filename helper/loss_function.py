import torch


def dice_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor):
    smooth = 1.

    i_flat = input_tensor.view(-1)
    t_flat = target_tensor.view(-1)
    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
