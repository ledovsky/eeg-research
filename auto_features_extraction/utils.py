import typing as tp

import torch
from torch import nn
from torch.utils.data import Dataset


class Permute(nn.Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(self.dims)


class Square(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input ** 2


class Log(nn.Module):
    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.clamp(min=self.eps).log()


class ShallowConvNet(nn.Sequential):
    def __init__(self,
                 num_channels,
                 num_classes,
                 num_units=40,
                 temporal_kernel_size=25,
                 pool_kernel_size=75,
                 pool_stride=15,
                 dropout_rate=0.5) -> None:
        super().__init__(
            nn.Unflatten(1, (1, num_channels)),
            nn.Conv2d(1, num_units, (1, temporal_kernel_size)),
            nn.Conv2d(num_units, num_units, (num_channels, 1)),
            Square(),
            nn.Flatten(2),
            nn.AvgPool2d((1, pool_kernel_size),
                         stride=(1, pool_stride)),
            Log(),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(1200, num_classes),
        )


class DeepConvNet(nn.Sequential):
    def __init__(self,
                 num_channels,
                 num_classes,
                 num_units=25,
                 temporal_kernel_size=10,
                 pool_kernel_size=3,
                 pool_stride=3,
                 dropout_rate=0.5) -> None:
        super().__init__(
            nn.Unflatten(1, (1, num_channels)),
            nn.Conv2d(1, num_units, (1, temporal_kernel_size)),
            nn.Conv2d(num_units, num_units, (num_channels, 1)),
            nn.ELU(),
            # nn.Flatten(2),
            nn.MaxPool2d((1, pool_kernel_size),
                         stride=(1, pool_stride)),
            nn.Conv2d(num_units, 2 * num_units,
                      (1, temporal_kernel_size)),
            nn.ELU(),
            nn.MaxPool2d((1, pool_kernel_size),
                         stride=(1, pool_stride)),
            nn.Conv2d(2 * num_units, 4 * num_units,
                      (1, temporal_kernel_size)),
            nn.ELU(),
            nn.MaxPool2d((1, pool_kernel_size),
                         stride=(1, pool_stride)),
            nn.Conv2d(4 * num_units, 8 * num_units,
                      (1, temporal_kernel_size)),
            nn.ELU(),
            nn.MaxPool2d((1, pool_kernel_size),
                         stride=(1, pool_stride)),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(400, num_classes),
        )


class EEGDataset(Dataset):
    def __init__(self,
                 features: tp.List[torch.Tensor],
                 labels: torch.Tensor,
                 transform=None):
        assert len(features) == labels.size(0)
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is None:
            return self.features[index], self.labels[index]
        return self.transform(self.features[index]), self.labels[index]

    def __len__(self):
        return len(self.features)
