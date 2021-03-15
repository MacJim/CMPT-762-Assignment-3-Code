import os
import unittest

import torch

from segmentation_unet import NestedUNet


class NestedUNetTestCase (unittest.TestCase):
    def test_network_input_output_shape(self):
        """
        Input and output have the same shape (N, C, H, W).
        """
        batch_size = 2
        in_channels = 3
        out_channels = 1
        # height, width = 512, 512    # The network only accepts specific input sizes.
        side_lengths = [512, 480, 384, 256, 240]
        failed_side_lengths = [500, 460, 440, 420, 400]    # Something wrong with the convolutions.

        model = NestedUNet(out_channels)
        model = model.cuda()

        for side_length in side_lengths:
            with self.subTest(side_length=side_length):
                x = torch.rand((batch_size, in_channels, side_length, side_length))    # (N, C, H, W)
                x = x.cuda()
                output = model(x)    # (N, C, H, W)
                # print(f"Input: {x.shape}, output: {output.shape}")
                del output    # Receives CUDA out of memory error without this line.


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    unittest.main()
