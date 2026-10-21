import math
import numpy as np
import os
import random
import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import DropPath
from basicsr_dctp.ops.fused_act import FusedLeakyReLU, fused_leaky_relu
from basicsr_dctp.ops.upfirdn2d import upfirdn2d
from basicsr_dctp.utils.registry import ARCH_REGISTRY

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# degradation encoder
from basicsr_dctp.archs.dual_encoder_arch import Encoder



class NormStyleCode(nn.Module):

    def forward(self, x):
        """Normalize the style codes.

        Args:
            x (Tensor): Style codes with shape (b, c).

        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.

    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.

    Returns:
        Tensor: 2D resampled kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]  # to 2D kernel, outer product
    # normalize
    k /= k.sum()
    return k


class UpFirDnUpsample(nn.Module):
    """Upsample, FIR filter, and downsample (upsampole version).

    References:
    1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html  # noqa: E501
    2. http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/signal/upfirdn.html  # noqa: E501

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Upsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnUpsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel) * (factor**2)
        self.factor = factor

        pad = self.kernel.shape[0] - factor
        self.pad = ((pad + 1) // 2 + factor - 1, pad // 2)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=self.factor, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(factor={self.factor})')


class UpFirDnDownsample(nn.Module):
    """Upsample, FIR filter, and downsample (downsampole version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Downsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnDownsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel)
        self.factor = factor

        pad = self.kernel.shape[0] - factor
        self.pad = ((pad + 1) // 2, pad // 2)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=self.factor, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(factor={self.factor})')


class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Deafult: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * (upsample_factor**2)

        if upsample_factor > 1:
            pad = (self.kernel.shape[0] - upsample_factor) - (kernel_size - 1)
            self.pad = ((pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1)
        elif downsample_factor > 1:
            pad = (self.kernel.shape[0] - downsample_factor) + (kernel_size - 1)
            self.pad = ((pad + 1) // 2, pad // 2)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}'
                f', downsample_factor={self.downsample_factor})')


class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f'Wrong activation value in EqualLinear: {activation}'
                             "Supported ones are: ['fused_lrelu', None].")
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, bias={self.bias is not None})')


class ModulatedConv2d(nn.Module):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1),
                 eps=1e-8):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        if self.sample_mode == 'upsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=2, downsample_factor=1, kernel_size=kernel_size)
        elif self.sample_mode == 'downsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size)
        elif self.sample_mode is None:
            pass
        else:
            raise ValueError(f'Wrong sample mode {self.sample_mode}, '
                             "supported ones are ['upsample', 'downsample', None].")

        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        # modulation inside each modulated conv
        self.modulation = EqualLinear(
            num_style_feat, in_channels, bias=True, bias_init_val=1, lr_mul=1, activation=None)

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).

        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.scale * self.weight * style  # (b, c_out, c_in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.sample_mode == 'upsample':
            x = x.view(1, b * c, h, w)
            weight = weight.view(b, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(b * c, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
            out = self.smooth(out)
        elif self.sample_mode == 'downsample':
            x = self.smooth(x)
            x = x.view(1, b * c, *x.shape[2:4])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        else:
            x = x.view(1, b * c, h, w)
            # weight: (b*c_out, c_in, k, k), groups=b
            out = F.conv2d(x, weight, padding=self.padding, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, '
                f'demodulate={self.demodulate}, sample_mode={self.sample_mode})')


class StyleConv(nn.Module):
    """Style conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1)):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.modulated_conv(x, style)
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # activation (with bias)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """To RGB from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, num_style_feat, upsample=True, resample_kernel=(1, 3, 3, 1)):
        super(ToRGB, self).__init__()
        if upsample:
            self.upsample = UpFirDnUpsample(resample_kernel, factor=2)
        else:
            self.upsample = None
        self.modulated_conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        """Forward function.

        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.

        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)
            out = out + skip
        return out

class ToRGB16(nn.Module):
    """To RGB from features.

    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, num_style_feat, upsample=True, resample_kernel=(1, 3, 3, 1)):
        super(ToRGB16, self).__init__()
        if upsample:
            self.upsample = UpFirDnUpsample(resample_kernel, factor=2)
        else:
            self.upsample = None
        self.modulated_conv = ModulatedConv2d(
            in_channels, 8, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None)
        self.bias = nn.Parameter(torch.zeros(1, 8, 1, 1))

    def forward(self, x, style, skip=None):
        """Forward function.

        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.

        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = self.upsample(skip)
            out = out + skip
        return out

class ConstantInput(nn.Module):
    """Constant input.

    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """

    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out

class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)

class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size},'
                f' stride={self.stride}, padding={self.padding}, '
                f'bias={self.bias is not None})')


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kenrel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 resample_kernel=(1, 3, 3, 1),
                 bias=True,
                 activate=True):
        layers = []
        # downsample
        if downsample:
            layers.append(
                UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        # conv
        layers.append(
            EqualConv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias
                and not activate))
        # activation
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super(ConvLayer, self).__init__(*layers)


class StyleGAN2Generator(nn.Module):
    """StyleGAN2 Generator.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1):
        super(StyleGAN2Generator, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    num_style_feat, num_style_feat, bias=True, bias_init_val=0, lr_mul=lr_mlp,
                    activation='fused_lrelu'))
        self.style_mlp = nn.Sequential(*style_mlp_layers)

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels['4'], size=4)
        self.style_conv1 = StyleConv(
            channels['4'],
            channels['4'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            resample_kernel=resample_kernel)
        self.to_rgb1 = ToRGB16(channels['4'], num_style_feat, upsample=False, resample_kernel=resample_kernel)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels['4']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample',
                    resample_kernel=resample_kernel,
                ))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                    resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB16(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(num_latent, self.num_style_feat, device=self.constant_input.weight.device)
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(self,
                styles,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2Generator.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latent with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.size(0))
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class Norm2Scale(nn.Module):
    def forward(self, scale1, scale2, scale3=None):
        if scale3 is not None:
            scales_norm = scale1**2 + scale2**2 + scale3**2 + 1e-8
            return scale1 * torch.rsqrt(scales_norm), scale2 * torch.rsqrt(scales_norm), scale3 * torch.rsqrt(scales_norm)
        else:
            scales_norm = scale1**2 + scale2**2 + 1e-8
            return scale1 * torch.rsqrt(scales_norm), scale2 * torch.rsqrt(scales_norm)

class StyleConv_norm_scale_shift(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1)):
        super(StyleConv_norm_scale_shift, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = FusedLeakyReLU(out_channels)
        self.norm = Norm2Scale()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 5

        self.avgpool = nn.AvgPool2d(2,2)

    def forward(self, x, style, noise=None, scale1=None, scale2=None, scale3=None, shift=None, swin=None):
    # def forward(self, x, style, noise=None, deg=None, shift=None):
        # modulate
        out = self.modulated_conv(x, style)
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        
        b, c, h, w = out.shape
        scale1, scale2 = self.norm(scale1, scale2, scale3)
        out = out * scale1.view(-1, out.size(1), 1, 1) + shift * scale2.view(-1, out.size(1), 1, 1)

        out = self.activate(out)
        return out
       

def grid_sample(x, offset, scale, scale2):
    # generate grids
    b, _, h, w = x.size()
    grid = np.meshgrid(range(round(scale2*w)), range(round(scale*h)))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale2 - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode='zeros')

    return output


class SA_upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False):
        super(SA_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, scale2):
        b, c, h, w = x.size()

        coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, round(w * scale2), 1).unsqueeze(0).float().to(x.device)]

        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale2) - (torch.floor((coor_hr[1] + 0.5) / scale2 + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale2,
            torch.ones_like(coor_h).expand([-1, round(scale2 * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, round(scale2 * w)]).unsqueeze(0),
            coor_w.expand([round(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)

        embedding = self.body(input)
        offset = self.offset(embedding)

        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale2*w)).transpose(0, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, round(scale*h), round(scale2*w), self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, round(scale*h), round(scale2*w), self.channels, self.channels//8)

        fea0 = grid_sample(x, offset, scale, scale2)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0


class ResBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kernel to 2D resample kernel.
            Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, out_channels, resample_kernel=(1, 3, 3, 1)):
        super(ResBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvLayer(
            in_channels, out_channels, 3, downsample=True, resample_kernel=resample_kernel, bias=True, activate=True)
        self.skip = ConvLayer(
            in_channels, out_channels, 1, downsample=True, resample_kernel=resample_kernel, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


@ARCH_REGISTRY.register()
class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    Args:
        out_size (int): The spatial size of outputs.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        stddev_group (int): For group stddev statistics. Default: 4.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), stddev_group=4, narrow=1):
        super(StyleGAN2Discriminator, self).__init__()

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }

        log_size = int(math.log(out_size, 2))

        conv_body = [ConvLayer(3, channels[f'{out_size}'], 1, bias=True, activate=True)]

        in_channels = channels[f'{out_size}']
        for i in range(log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            conv_body.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_body)

        self.final_conv = ConvLayer(in_channels + 1, channels['4'], 3, bias=True, activate=True)
        self.final_linear = nn.Sequential(
            EqualLinear(
                channels['4'] * 4 * 4, channels['4'], bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'),
            EqualLinear(channels['4'], 1, bias=True, bias_init_val=0, lr_mul=1, activation=None),
        )
        self.stddev_group = stddev_group
        self.stddev_feat = 1

    def forward(self, x):
        out = self.conv_body(x)

        b, c, h, w = out.shape
        # concatenate a group stddev statistics to out
        group = min(b, self.stddev_group)  # Minibatch must be divisible by (or smaller than) group_size
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(b, -1)
        out = self.final_linear(out)

        return out


@ARCH_REGISTRY.register()
class StyleGAN2Discriminator_no_std(nn.Module):
    """StyleGAN2 Discriminator.

    Args:
        out_size (int): The spatial size of outputs.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kenrel to 2D resample kernel. Default: (1, 3, 3, 1).
        stddev_group (int): For group stddev statistics. Default: 4.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), stddev_group=4, narrow=1):
        super(StyleGAN2Discriminator_no_std, self).__init__()

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }

        log_size = int(math.log(out_size, 2))

        conv_body = [ConvLayer(3, channels[f'{out_size}'], 1, bias=True, activate=True)]

        in_channels = channels[f'{out_size}']
        for i in range(log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            conv_body.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_body)

        self.final_conv = ConvLayer(in_channels, channels['4'], 3, bias=True, activate=True)
        self.final_linear = nn.Sequential(
            EqualLinear(
                channels['4'] * 4 * 4, channels['4'], bias=True, bias_init_val=0, lr_mul=1, activation='fused_lrelu'),
            EqualLinear(channels['4'], 1, bias=True, bias_init_val=0, lr_mul=1, activation=None),
        )
    def forward(self, x):

        out = self.conv_body(x) # 2, 512, 4, 4

        b, c, h, w = out.shape

        out = self.final_conv(out) # 2, 512, 4, 4
        out = out.view(b, -1)
        out = self.final_linear(out) # 2, 1
        return out


class Style_fuse_module(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1)):
        super(Style_fuse_module, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = FusedLeakyReLU(out_channels)
        self.norm = Norm2Scale()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 5

        self.avgpool = nn.AvgPool2d(2,2)

    def forward(self, x, style, noise=None, scale1=None, scale2=None, shift=None):
        out = self.modulated_conv(x, style)
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        
        b, c, h, w = out.shape
        scale1, scale2 = self.norm(scale1, scale2)
        out = out * scale1.view(-1, out.size(1), 1, 1) + shift * scale2.view(-1, out.size(1), 1, 1)

        out = self.activate(out.view(b, c, h, w))
        return out




class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.3):

        super(Conv2d_cd, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.theta = torch.nn.Parameter(torch.FloatTensor([0.3]), requires_grad=True)
       
    def forward(self, x):
        pad_x = self.reflection_pad(x)
        out_normal = self.conv(pad_x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class PatchEmbed(nn.Module):
    """ Texture Embedding """
    def __init__(self, input_dim=32, embed_dim=32, norm_layer=None):
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TextureAttetion(nn.Module):
    """ Texture Attention. """
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads or dim
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv_texture = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, texture_map):
        B, N, C = x.shape
        k = self.qkv(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        qkv_t = self.qkv_texture(texture_map).reshape(-1, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_t, v_t = qkv_t.unbind(0)
        # trick here to make q@k.t more stable
        attn = (q_t.repeat(k.shape[0]//q_t.shape[0], 1, 1, 1) * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v_t.repeat(k.shape[0]//q_t.shape[0], 1, 1, 1)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """ feed-forward network(MLP) """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class TextureTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn = PreNorm(dim, TextureAttetion(dim, num_heads = num_heads, attn_drop = dropout))
        self.ff = PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), dropout = dropout))  # MLP

    def forward(self, x, texture_map):
        x = self.attn(x, texture_map=texture_map) + x
        x = self.ff(x) + x
        return x


class StructureTransformerBlock(nn.Module):
    def __init__(self, dim, embed_dim, num_heads, mlp_ratio = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.embedding = PatchEmbed(dim, embed_dim, norm_layer=None)
        self.attn = PreNorm(embed_dim, TextureAttetion(embed_dim, num_heads = num_heads, attn_drop = dropout))
        self.ff = PreNorm(embed_dim, FeedForward(embed_dim, dim * mlp_ratio, dropout = dropout))  # MLP

    def forward(self, x):
        x = self.embedding(x)
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Stage(nn.Module):
    """ A basic 3D Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: False
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, out_channels, window_size=16,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.texture_map = nn.Parameter(torch.Tensor(1, (input_resolution//window_size)**2, dim//window_size)).cuda()
        nn.init.kaiming_normal_(self.texture_map)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (c p1) (h p2) (w p3) -> b (h w) (p1 p2 p3 c)', p1=window_size, p2=window_size, p3=window_size),
            nn.Linear(dim*window_size*window_size, dim//window_size)
        )

        self.texture_transformer = TextureTransformerBlock(dim//window_size, window_size // 32 or 1, mlp_ratio=mlp_ratio)
       

        self.unembedding = nn.Sequential(
            nn.Linear(dim//window_size, dim*window_size*window_size),
            Rearrange('b (h w) (p1 p2 p3 c) -> b (c p1) (h p2) (w p3)', h=input_resolution//window_size, w=input_resolution//window_size, p1=window_size, p2=window_size, p3=window_size)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.to_patch_embedding(x)
       
        x = self.texture_transformer(x, self.texture_map)
        x = self.unembedding(x)

        return x




class Style_Multi_fuse_module(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_channels,
                 down_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1)):
        super(Style_Multi_fuse_module, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            resample_kernel=resample_kernel)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = FusedLeakyReLU(out_channels)
        self.norm = Norm2Scale()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 5

        if up_channels is not None:
            self.down = nn.ConvTranspose2d(up_channels, in_channels, 2, 2)

        if down_channels is not None:
            self.up = ConvLayer(down_channels, in_channels, 3, downsample=True)


    def forward(self, x, style, noise=None, scale=None, shift=None, shift_front=None, shift_after=None):
        out = self.modulated_conv(x, style)
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        
        b, c, h, w = out.shape

        if shift_after is None:
            scale1, scale2 = self.norm(*scale.view(scale.shape[0], 2, -1).unbind(1))
            front = self.down(shift_front)
            out = front * scale1.view(b, out.size(1), 1, 1) + out * scale2.view(b, out.size(1), 1, 1)

        elif shift_front is None:
            scale1, scale2 = self.norm(*scale.view(scale.shape[0], 2, -1).unbind(1))
            after = self.up(shift_after)
            out = out * scale1.view(b, out.size(1), 1, 1) + after * scale2.view(b, out.size(1), 1, 1)

        else:
            scale1, scale2, scale3 = self.norm(*scale.view(scale.shape[0], 3, -1).unbind(1))
            front = self.down(shift_front)
            after = self.up(shift_after)
            out = front * scale1.view(b, out.size(1), 1, 1) + out * scale2.view(b, out.size(1), 1, 1) + after * scale3.view(-1, out.size(1), 1, 1)

        out = self.activate(out.view(b, c, h, w))
        return out


    

@ARCH_REGISTRY.register()
class DCTP(nn.Module):

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 narrow=1):
        super(DCTP, self).__init__()

        self.num_style_feat = num_style_feat

        # ------------ 定义退化编码器 ------------ #
        self.content_encoder = Encoder()
        self.degradation_encoder = Encoder()
        content_encoder_path = "./weights/content.pt"
        degradation_encoder_path = "./weights/degradation.pt"
        if os.path.isfile(content_encoder_path):
            self.content_encoder.load_state_dict(torch.load(content_encoder_path), strict=True)
            for param in self.content_encoder.parameters():
                param.requires_grad = False
        if os.path.isfile(degradation_encoder_path):
            self.degradation_encoder.load_state_dict(torch.load(degradation_encoder_path), strict=True)
            for param in self.degradation_encoder.parameters():
                param.requires_grad = False
        # ------------------------------------- #

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }
        self.channels = channels

        self.log_size = int(math.log(out_size, 2))
        self.num_latent = (self.log_size - 2) * 2 - 2

        first_out_size = 2**(int(math.log(out_size, 2)))

        self.encoder_channels = channels

        self.conv_body_first = ConvLayer(3, channels[f'{first_out_size}'], 3, bias=True, activate=True)
        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        # self.only_down = nn.ModuleList()
        # self.condition_encoder = nn.ModuleList()
        for i in range(self.log_size-1, 2+1, -1):
            out_channels = channels[f'{2**i}']
            self.conv_body_down.append(ConvLayer(in_channels, out_channels, 3, downsample=True))
            # self.only_down.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1))
            # self.condition_encoder.append(EqualLinear(4, out_channels, bias=True, bias_init_val=1, activation=None))
            in_channels = out_channels
        self.final_conv = ConvLayer(channels['16'], channels['16'], 3)

        self.final_down1 = ConvLayer(channels['16'], channels['8'], 3, downsample=True)
        self.final_down2 = ConvLayer(channels['8'], channels['4']//2, 3, downsample=True)
        self.final_linear = EqualLinear(2 * 4 * 512, self.num_style_feat * (self.num_latent+2), bias=True, activation='fused_lrelu')

        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()

        for i in range(self.log_size, 2+1, -1):
            #print("1")
            out_channels = channels[f'{2**i}']
            in_channels = channels[f'{2**i}']

            if i == self.log_size or i == 4:
                self.condition_scale.append(
                    EqualLinear(256, out_channels * 2, bias=True, bias_init_val=1, activation=None))
            else:
                self.condition_scale.append(
                    EqualLinear(256, out_channels * 3, bias=True, bias_init_val=1, activation=None))
           
            self.condition_shift.append(
                ConvLayer(in_channels, out_channels, 3, bias=True, activate=False))

        # stylegan decoder
        self.style_conv1 = Style_Multi_fuse_module(
            channels['16'],
            channels['16'],
            up_channels = None,
            down_channels = channels['32'],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            resample_kernel=resample_kernel)

        self.to_rgb1 = ToRGB(channels['16'], num_style_feat, upsample=False, resample_kernel=resample_kernel)
        
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2 - 2) * 2 + 1

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.texture_conv = nn.ModuleList()
        self.cdc_conv = nn.ModuleList()
        self.tDTransformer = nn.ModuleList()

        self.noises = nn.Module()

        in_channels = channels['16']
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2**((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f'noise{layer_idx}', torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3+2, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample',
                    resample_kernel=resample_kernel,
                ))
            if i != self.log_size:
                self.style_convs.append(
                    Style_Multi_fuse_module(
                        out_channels,
                        out_channels,
                        up_channels = channels[f'{2**(i-1)}'],
                        down_channels = channels[f'{2**(i+1)}'],
                        kernel_size=3,
                        num_style_feat=num_style_feat,
                        demodulate=True,
                        sample_mode=None,
                        resample_kernel=resample_kernel))
            else:
                self.style_convs.append(
                    Style_Multi_fuse_module(
                        out_channels,
                        out_channels,
                        up_channels = channels[f'{2**(i-1)}'],
                        down_channels = None,
                        kernel_size=3,
                        num_style_feat=num_style_feat,
                        demodulate=True,
                        sample_mode=None,
                        resample_kernel=resample_kernel))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True, resample_kernel=resample_kernel))
            if i >= 8:
                self.texture_conv.append(
                    StyleConv(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        num_style_feat=num_style_feat,
                        demodulate=True,
                        sample_mode='upsample',
                        resample_kernel=resample_kernel,
                    ))
                self.tDTransformer.append(
                    Stage(out_channels, 2**i, out_channels // 32, out_channels)
                )
                self.cdc_conv.append(Conv2d_cd(out_channels, out_channels))
               
            in_channels = out_channels
        self.norm = Norm2Scale()
        # self.linear_alpha = EqualLinear(1, out_channels * 2, bias=True, bias_init_val=1, activation=None)
        self.linear_alpha1 = EqualLinear(1, out_channels * 2 * 2, bias=True, bias_init_val=1, activation=None)
        self.linear_alpha2 = EqualLinear(1, out_channels * 2, bias=True, bias_init_val=1, activation=None)
        self.gama = 5
        print("Init DCTP")
        


    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises
    
    # scale:
    #[[4, 128], [4, 384], [4, 768], [4, 1536], [4, 1536], [4, 1024]]

    def forward(self, x1, degradation, alpha=None, noise=None,
                randomize_noise=True):
        
        #----------- 获取退化表示 -----------#
        self.content_encoder.eval()
        self.degradation_encoder.eval()
        d_content = self.content_encoder(x1)
        d_degradation = self.degradation_encoder(x1)
        degradation = d_degradation - self.gama * d_content
        #degradation = d_degradation

        #print(d_degradation.shape,d_content.shape,degradation.shape)


        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]

        b, c, h, w = x1.shape
        # main generation
        feat1 = self.conv_body_first(x1) # 1, 32, 1024, 1024
        scales, shifts = [], []

        scale = self.condition_scale[0](degradation)
        scales.append(scale.clone())
        # scale2 = self.condition_scale2[0](degradation)
        # scales2.append(scale2.clone())

        shift = self.condition_shift[0](feat1) # 1, 32, 1024, 1024
        shifts.append(shift.clone())

        j = 1
        for i in range(len(self.conv_body_down)):
            feat1 = self.conv_body_down[i](feat1)
            # 1, 64, 512, 512
            # 1, 128, 256, 256
            # 1, 256, 128, 128
            # 1, 512, 64, 64
            # 1, 512, 32, 32
            # 1, 512, 16, 16
            if j < len(self.condition_scale):
                scale = self.condition_scale[j](degradation)
                scales.append(scale.clone())
                # scale2 = self.condition_scale2[j](degradation)
                # scales2.append(scale2.clone())
                shift = self.condition_shift[j](feat1)
                shifts.append(shift.clone())
                j += 1
        
        scales = scales[::-1]
        # scales2 = scales2[::-1]
        shifts = shifts[::-1]

        b = feat1.size(0)  # feat: 1, 512, 16, 16

        tmp1 = self.final_down2(self.final_down1(feat1)) # 1, 256, 4, 4
        latent1 = self.final_linear(tmp1.view(b, -1)).view(-1, self.num_latent+2, self.num_style_feat)

        # 在这加swin
        # sw_out = self.final_swin(feat.flatten(2).transpose(1, 2), (16, 16))
        out = self.final_conv(feat1) # 1, 512, 16, 16
        out = self.style_conv1(out, latent1[:, 0], noise=noise[0], scale=scales[0], shift=shifts[0], shift_front=None, shift_after=shifts[1])
        skip = self.to_rgb1(out, latent1[:, 1])

        i = 1
        j = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            if j==4:
                texture_out = self.texture_conv[j-4](out, latent1[:, j + 7])
                # 3D transformer
                texture_out = self.tDTransformer[j-4](texture_out)
                # cdc
                texture_out = self.cdc_conv[j-4](texture_out)
                scale = self.linear_alpha1(alpha).view(b, 2, -1)

                # scale_alpha1, scale_alpha2 = self.softmax(scale.view(b, 2, -1)).unbind(1)

                # scale_alpha2 = self.linear_alpha2[j-4](alpha)
                scale_alpha1, scale_alpha2 = self.norm(*scale.view(b, 2, -1).unbind(1))

                out = conv1(out, latent1[:, i], noise=noise1)
                out = conv2(out, latent1[:, i + 1], noise=noise2, scale=scales[j], shift=shifts[j], shift_front=shifts[j-1], shift_after= None if j+1 >= len(shifts) else shifts[j+1])
                out = out * scale_alpha1.view(b, out.size(1), 1, 1) + texture_out * scale_alpha2.view(b, texture_out.size(1), 1, 1)

                skip = to_rgb(out, latent1[:, i + 2], skip)
            elif j==5:
                texture_out = self.texture_conv[j-4](texture_out, latent1[:, j + 7])
                # 3D transformer
                texture_out = self.tDTransformer[j-4](texture_out)
                # cdc
                texture_out = self.cdc_conv[j-4](texture_out)
                scale = self.linear_alpha2(alpha).view(b, 2, -1)

                # scale_alpha1, scale_alpha2 = self.softmax(scale.view(b, 2, -1)).unbind(1)

                # scale_alpha2 = self.linear_alpha2[j-4](alpha)
                scale_alpha1, scale_alpha2 = self.norm(*scale.view(b, 2, -1).unbind(1))

                out = conv1(out, latent1[:, i], noise=noise1)
                out = conv2(out, latent1[:, i + 1], noise=noise2, scale=scales[j], shift=shifts[j], shift_front=shifts[j-1], shift_after= None if j+1 >= len(shifts) else shifts[j+1])
                out = out * scale_alpha1.view(b, out.size(1), 1, 1) + texture_out * scale_alpha2.view(b, texture_out.size(1), 1, 1)
                skip = to_rgb(out, latent1[:, i + 2], skip)
            else:
                out = conv1(out, latent1[:, i], noise=noise1)
                out = conv2(out, latent1[:, i + 1], noise=noise2, scale=scales[j], shift=shifts[j], shift_front=shifts[j-1], shift_after=shifts[j+1])
            
                skip = to_rgb(out, latent1[:, i + 2], skip)

            i += 2
            j += 1

        image = skip
        return image, latent1
    

    
    




