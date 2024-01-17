import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ECGNetwork(nn.Module):
    def __init__(self, n_channels):
        super(ECGNetwork, self).__init__()
        # TODO: Create proper network
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * reduced_length, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
