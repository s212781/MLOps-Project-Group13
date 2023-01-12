import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dims = 100
        y = lambda a: torch.floor((a - 4) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * y(y(y(self.input_dims))), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
