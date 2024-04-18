from .modules import (
    Swish,
    MemoryEfficientSwish,
    PreActivatedConv3dReLU,
    PointWiseConv3d,
    DepthWiseConv3d,
    Attention,
    SCSEModule,
)

from .utils import(
    get_same_padding_conv3d,
    drop_connect,
    calculate_output_image_size,
)

import torch
from torch import nn
from torch.nn import functional as F


class MBConvBlock3D(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
        decoder_mode (bool): Reverse the block (deconvolution) if true.

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None, decoder_mode=False, decoder_output_image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
        self.decoder_mode = decoder_mode

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv3d = get_same_padding_conv3d(image_size=image_size, transposed=self.decoder_mode)
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        if self.decoder_mode:
            # assert decoder_output_image_size
            image_size = decoder_output_image_size
        Conv3d = get_same_padding_conv3d(image_size=image_size, transposed=self.decoder_mode)
        self._depthwise_conv = Conv3d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if not self.decoder_mode:
            image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv3d = get_same_padding_conv3d(image_size=(1, 1, 1), transposed=self.decoder_mode)
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv3d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv3d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv3d = get_same_padding_conv3d(image_size=image_size, transposed=self.decoder_mode)
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm3d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock3D's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class UnetCenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = PreActivatedConv3dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = PreActivatedConv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
            dropout=None,
    ):
        super().__init__()

        self.dropout = nn.Dropout3d(dropout) if dropout else nn.Identity()

        self.conv1 = PreActivatedConv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = PreActivatedConv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.dropout(x)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.attention2(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted bottleneck residual block with an scSE block embedded into the residual layer, after the 
    depthwise convolution. By default, uses batch normalization and Hardswish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, expansion_ratio = 1, squeeze_ratio = 1, \
        activation = nn.Hardswish(True), normalization = nn.BatchNorm3d):
        super().__init__()
        self.same_shape = in_channels == out_channels
        self.mid_channels = expansion_ratio*in_channels
        self.block = nn.Sequential(
            PointWiseConv3d(in_channels, self.mid_channels),
            normalization(self.mid_channels),
            activation,
            DepthWiseConv3d(self.mid_channels, kernel_size=kernel_size, stride=stride),
            normalization(self.mid_channels),
            activation,
            #md.sSEModule(self.mid_channels),
            SCSEModule(self.mid_channels, reduction = squeeze_ratio),
            #md.SEModule(self.mid_channels, reduction = squeeze_ratio),
            PointWiseConv3d(self.mid_channels, out_channels),
            normalization(out_channels)
        )
        
        if not self.same_shape:
            # 1x1 convolution used to match the number of channels in the skip feature maps with that 
            # of the residual feature maps
            self.skip_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                normalization(out_channels)
            )
          
    def forward(self, x):
        residual = self.block(x)
        
        if not self.same_shape:
            x = self.skip_conv(x)
        return x + residual


class UnetPlusPlusDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            squeeze_ratio=1,
            expansion_ratio=1
    ):
        super().__init__()

        # Inverted Residual block convolutions
        self.conv1 = InvertedResidual(
            in_channels=in_channels+skip_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )
        self.conv2 = InvertedResidual(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
