"""Gaussian heads. GaussianXYZHead matches the published Allen transfer MLP."""

import torch
from torch import nn


class GaussianXYZHead(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mu_head = nn.Linear(128, 3)
        self.logvar_head = nn.Linear(128, 3)

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.logvar_head(h).clamp(-10.0, 10.0)


class LinearGaussianHead(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.mu_head = nn.Linear(input_dim, 3)
        self.logvar_head = nn.Linear(input_dim, 3)

    def forward(self, x):
        return self.mu_head(x), self.logvar_head(x).clamp(-10.0, 10.0)


def gaussian_nll(mu, logvar, target):
    return 0.5 * (logvar + (target - mu).square() * torch.exp(-logvar)).mean()
