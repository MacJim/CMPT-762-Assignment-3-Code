import torch


def dice_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor):
    smooth = 1.

    # i_flat = input_tensor.view(-1)
    # t_flat = target_tensor.view(-1)
    intersection = (input_tensor * target_tensor).sum()

    return 1 - ((2. * intersection + smooth) / (input_tensor.sum() + target_tensor.sum() + smooth))
