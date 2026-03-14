import torch
from torch import nn


test = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
print(test.weight.shape)