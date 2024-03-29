"""
Models code from speechbrain
(https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/)
"""

import math

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from lib.layers.norm import BatchNorm1d as MLXBathNorm1d
from mlx.nn import Conv1d as MLXConv1d, ReLU as MLXReLU
from lib.layers.conv import Conv1d as ConvLayer, BatchNorm1d as BatchNormLayer
from typing import Any


def length_to_mask_mx(length, max_len=None):
    assert len(length.shape) == 1
    if max_len is None:
        max_len = mx.max(length)
    mask = mx.arange(max_len)[None, :] < length[:, None]
    return mask


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding


# Dummy
class Tanh(nn.Module):
    def __init__(self):
        super().__init__()


# Dummy
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()


class Conv1d(nn.Module):
    """This function implements 1d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : int
        Kernel size of the convolutional filters.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride : int
        Stride factor of the convolutional filters. When the stride factor > 1,
        a decimation in time is performed.
    dilation : int
        Dilation factor of the convolutional filters.
    padding : str
        (same, valid, causal). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is the same as the input shape.
        "causal" results in causal (dilated) convolutions.
    groups: int
        Number of blocked connections from input channels to output channels.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    skip_transpose : bool
        If False, uses batch x time x channel convention of speechbrain.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference
    conv_init : str
        Weight initialization for the convolution network
    default_padding: str or int
        This sets the default padding mode that will be used by the pytorch Conv1d backend.

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16])
    >>> cnn_1d = Conv1d(
    ...     input_shape=inp_tensor.shape, out_channels=8, kernel_size=5
    ... )
    >>> out_tensor = cnn_1d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 8])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=1,
        dilation=1,
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=True,
        weight_norm=False,
        conv_init=None,
        default_padding=0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input_shape(input_shape)

        self.in_channels = in_channels

        self.conv = MLXConv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=default_padding,
            bias=bias,
        )

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.
        """
        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size, self.dilation, self.stride)

        elif self.padding == "causal":
            num_pad = (self.kernel_size - 1) * self.dilation
            padding_spec = [(0, 0)] * (x.ndim) + [(num_pad, 0)]
            x = mx.pad(x, padding_spec, mode="constant", constant_values=0)

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got " + self.padding
            )

        x = mx.transpose(x, (0, 2, 1))
        wx = self.conv(x)
        wx = mx.transpose(wx, (0, 2, 1))
        return wx

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def _manage_padding(
        self,
        x,
        kernel_size: int,
        dilation: int,
        stride: int,
    ):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.
        """
        # Detecting input shape
        L_in = self.in_channels
        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)
        # Applying padding
        x = mx.pad(x, padding)
        return x

    def _check_input_shape(self, shape):
        """Checks the input shape and returns the number of input channels."""

        if len(shape) == 2:
            self.unsqueeze = True
            in_channels = 1
        elif self.skip_transpose:
            in_channels = shape[1]
        elif len(shape) == 3:
            in_channels = shape[2]
        else:
            raise ValueError("conv1d expects 2d, 3d inputs. Got " + str(len(shape)))

        # Kernel size must be odd
        if not self.padding == "valid" and self.kernel_size % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    # def remove_weight_norm(self):
    #     """Removes weight normalization at inference if used during training."""
    #     self.conv = nn.utils.remove_weight_norm(self.conv)


class BatchNorm1d(nn.Module):
    """Applies 1d batch normalization to the input tensor.

    Arguments
    ---------
    input_shape : tuple
        The expected shape of the input. Alternatively, use ``input_size``.
    input_size : int
        The expected size of the input. Alternatively, use ``input_shape``.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    momentum : float
        It is a value used for the running_mean and running_var computation.
    affine : bool
        When set to True, the affine parameters are learned.
    track_running_stats : bool
        When set to True, this module tracks the running mean and variance,
        and when set to False, this module does not track such statistics.
    combine_batch_time : bool
        When true, it combines batch an time axis.


    Example
    -------
    >>> input = torch.randn(100, 10)
    >>> norm = BatchNorm1d(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 10])
    """

    def __init__(
        self,
        input_shape=None,
        input_size=None,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        combine_batch_time=False,
        skip_transpose=False,
    ):
        super().__init__()
        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = MLXBathNorm1d(
            num_features=input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=False,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, [channels])
            input to normalize. 2d or 3d tensors are expected in input
            4d tensors can be used when combine_dims=True.
        """
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[3], shape_or[2])

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.

    Arguments
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        The number of output channels.
    kernel_size : int
        The kernel size of the TDNN blocks.
    dilation : int
        The dilation of the TDNN block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
        The groups size of the TDNN blocks.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.activation = MLXReLU()
        self.norm = BatchNormLayer(out_channels)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        x = self._manage_padding(x, self.kernel_size, 1, 1)
        x = mx.transpose(x, (0, 2, 1))
        out = self.conv(x)
        out = self.activation(out)
        out = mx.transpose(out, (0, 2, 1))
        out = self.norm(out)
        return out

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def _manage_padding(
        self,
        x,
        kernel_size: int,
        dilation: int,
        stride: int,
    ):
        """This function performs zero-padding on the time axis
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        kernel_size : int
            Size of kernel.
        dilation : int
            Dilation used.
        stride : int
            Stride.
        """

        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)

        # Applying padding
        padding_spec = [(0, 0)] * (x.ndim - 1) + [padding]
        x = mx.pad(x, padding_spec)

        return x


class Res2NetBlock(nn.Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = [
            TDNNBlock(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for i in range(scale - 1)
        ]
        self.scale = scale

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        y = []
        x_chunks = mx.split(x, self.scale, axis=1)
        for i, x_i in enumerate(x_chunks):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = mx.concatenate(y, axis=1)
        return y

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> se_layer = SEBlock(64, 16, 64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = se_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = MLXConv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = MLXReLU()
        self.conv2 = MLXConv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = Sigmoid()

    def forward(self, x, lengths=None):
        """Processes the input tensor x and returns an output tensor."""
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask_mx(lengths * L, max_len=L)
            mask = mx.expand_dims(mask, axis=1)
            total = mx.sum(mask, axis=2, keepdims=True)
            s = mx.sum(x * mask, axis=2, keepdims=True) / total
        else:
            s = mx.mean(x, axis=2, keepdims=True)

        s = mx.transpose(s, (0, 2, 1))
        s = self.relu(self.conv1(s))
        s = mx.sigmoid(self.conv2(s))
        s = mx.transpose(s, (0, 2, 1))

        out = s * x
        return out

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = AttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = Tanh()
        self.conv = ConvLayer(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = mx.sum(m * x, axis=dim)
            std = mx.sqrt(
                mx.clip(
                    mx.sum(m * (x - mx.expand_dims(mean, axis=dim) ** 2), axis=dim),
                    a_min=eps,
                    a_max=None,
                )
            )
            return mean, std

        if lengths is None:
            lengths = mx.ones((x.shape[0]))

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask_mx(lengths * L, max_len=L)
        mask = mx.expand_dims(mx.array(mask), axis=1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            total = mx.sum(mask, axis=2, keepdims=True)
            mean, std = _compute_statistics(x, mask / total)
            mean = repeat(mx.expand_dims(mean, axis=2), L, axis=2)
            std = repeat(mx.expand_dims(std, axis=2), L, axis=2)
            attn = mx.concatenate([x, mean, std], axis=1)
        else:
            attn = x

        # attn is numpy array
        tdnn_out = self.tdnn.forward(attn)
        # # Apply layers
        ftr = mx.tanh(tdnn_out)
        ftr = mx.transpose(ftr, (0, 2, 1))
        attn = self.conv(ftr)
        attn = mx.transpose(attn, (0, 2, 1))

        # Filter out zero-paddings
        # Keep numpy here for np.inf, "math.inf" is slow
        attn = mx.where(mask == 0, -np.inf, attn)
        attn = mx.exp(attn) / mx.sum(mx.exp(attn), axis=2, keepdims=True)
        mean, std = _compute_statistics(x, attn)

        # Append mean and std of the batch
        pooled_stats = mx.concatenate((mean, std), axis=1)
        pooled_stats = mx.expand_dims(pooled_stats, axis=2)

        return pooled_stats

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


def repeat(x, L, axis):
    return mx.concatenate([x] * L, axis=axis)


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    groups: int
    Number of blocked connections from input channels to output channels.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = MLXConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        """Processes the input tensor x and returns an output tensor."""
        residual = x
        if self.shortcut:
            x = mx.array(x.tolist())
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class ECAPA_TDNN(nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    groups : list of ints
        List of groups for kernels in each layer.

    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size,
        device="mps",
        lin_neurons=192,
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = []

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilation=dilations[0],
                groups=groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-2] * (len(channels) - 2),
            channels[-1],
            kernel_sizes[-1],
            dilation=dilations[-1],
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )

        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2, skip_transpose=True)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = mx.transpose(x, (0, 2, 1))

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = mx.concatenate(xl[1:], axis=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)

        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = mx.transpose(x, (0, 2, 1))
        return x

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


def load_embedding_model(
    file_path: str = "embedding_model.ckpt",
) -> ECAPA_TDNN:
    from mlx.utils import tree_map
    import torch

    ecapa_tdnn = ECAPA_TDNN(
        80,  # input_size
        lin_neurons=192,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
    )

    params = torch.load(file_path, map_location="cpu")
    params = tree_map(lambda p: mx.array(p.numpy(), dtype=mx.float32), params)
    # TODO: model.update(params) doesn't work and so model.load_weights() will also not work
    # set params manually
    for k, v in params.items():
        ecapa_tdnn[k] = v

    return ecapa_tdnn


if __name__ == "__main__":
    ecapa_tdnn = load_embedding_model()
    # ecapa_tdnn = ECAPA_TDNN(
    #     80,  # input_size
    #     lin_neurons=192,
    #     channels=[1024, 1024, 1024, 1024, 3072],
    #     kernel_sizes=[5, 3, 3, 3, 1],
    #     dilations=[1, 2, 3, 4, 1],
    #     attention_channels=128,
    # )
    # test example from audio pipeline
    features = mx.random.normal([2048, 501, 80])
    wav_lens = mx.random.normal([2048])

    # features = mx.random.normal([1, 4, 80])
    # wav_lens = mx.random.normal([1

    import time

    for _ in range(10):
        start = time.time()
        embeddings = ecapa_tdnn.forward(features, wav_lens)
        end = time.time()
        print("embeddings.shape: ", embeddings.shape)
        print("time taken: ", end - start)
        # assert embeddings.shape == [64, 1, 192]
        # clear from mps device, otherwise it will throw an memory error
        # del embeddings
