import mlx.nn as nn
from mlx.nn import Conv1d as MLXConv1d
from .norm import BatchNorm1d as MLXBathNorm1d


# Indirection to match the model specification
class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = MLXConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def __call__(self, x):
        return self.conv(x)


# Indirection to match the model specification
class BatchNorm1d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.norm = MLXBathNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def __call__(self, x):
        return self.norm(x)
