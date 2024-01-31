import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, input_channel_size: int, output_channel_size: int):
        def conv_block(input_channel_size: int, output_channel_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(input_channel_size, output_channel_size, kernel_size=3, padding=1),
                nn.BatchNorm1d(output_channel_size),
                nn.SiLU(),
            )

        super(_ResidualBlock, self).__init__()
        self.conv1 = conv_block(input_channel_size, output_channel_size)
        self.conv2 = conv_block(output_channel_size, output_channel_size)
        self.conv3 = conv_block(output_channel_size, output_channel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        skip = x
        x = self.conv2(x)
        x = self.conv3(x)

        return F.dropout1d(x + skip, 0.1)


class ECGNetwork(nn.Module):
    def __init__(self, input_channels: int, n_classes: int):
        super(ECGNetwork, self).__init__()
        self.block1 = _ResidualBlock(input_channels, 8)
        self.block2 = _ResidualBlock(8, 16)
        self.block3 = _ResidualBlock(16, 32)
        self.block4 = _ResidualBlock(32, 64)
        self.block5 = _ResidualBlock(64, 64)
        self.block6 = _ResidualBlock(64, 128)
        self.block7 = _ResidualBlock(128, 128)
        self.final = nn.Conv1d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return self.final(x)
