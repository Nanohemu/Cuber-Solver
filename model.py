import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            # 3 * 92 * 76
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 32 * 22 * 18
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 64 * 10 * 8
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            # 64 * 8 * 6
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            # 64 * 8 * 6 = 3072
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            # 1024
            nn.Linear(1024, 512),
            nn.ReLU(),
            # 512
            nn.Linear(512, n_actions)
            # n_actions
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
