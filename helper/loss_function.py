import torch
import torch.nn.functional as F


def dice_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor, smooth=1.0):
    """
    `input_tensor` and `target_tensor` must have the same shape.x
    """
    intersection = (input_tensor * target_tensor).sum()

    return 1 - ((2. * intersection + smooth) / (input_tensor.sum() + target_tensor.sum() + smooth))


def binary_cross_entropy_dice_loss(input_tensor: torch.Tensor, target_tensor: torch.Tensor):
    """
    - `input_tensor`: It only has a single channel: (N, 1, H, W)
    - `target_tensor` (N, H, W)
    """
    if (input_tensor.shape[1] == 1):
        # Remove the channel dimension.
        input_tensor = torch.squeeze(input_tensor, 1)

    bce = F.binary_cross_entropy_with_logits(input_tensor, target_tensor)

    input_tensor = torch.sigmoid(input_tensor)
    dice = dice_loss(input_tensor, target_tensor, smooth=0.00001)

    return bce / 2 + dice
