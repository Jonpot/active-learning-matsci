from .blocks import MBConvBlock3D
from .decoders import EfficientUnet3DDecoder, EfficientUnetPlusPlus3DDecoder
from .heads import SegmentationHead, ClassificationHead
from .initialization import initialize_decoder, initialize_head
from .modules import Swish, MemoryEfficientSwish
from .utils import(
    get_model_params,
    efficientnet3D_params,
    get_same_padding_conv3d,
    patch_first_conv,
    round_filters,
    round_repeats,
    calculate_output_image_size,
    _get_pretrained_settings
)

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from typing import Optional, Union, List


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class EfficientNet3D(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
        in_channels (int): Input data's channel number.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet_pytorch_3d import EfficientNet3D
        >>> inputs = torch.rand(1, 3, 224, 224, 224)
        >>> model = EfficientNet3D.from_name('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None, in_channels=3):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        self._bn_mom = 1 - self._global_params.batch_norm_momentum
        self._bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv3d = get_same_padding_conv3d(image_size=image_size)

        # Stem
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        self._blocks_image_size = [image_size]
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            self._blocks_image_size.append(image_size)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv3d = get_same_padding_conv3d(image_size=image_size)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=self._bn_mom, eps=self._bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        
        # Set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

        # For autoencoder
        self._image_size = image_size
        self._last_block_args = block_args
        self._last_out_channels = out_channels

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                    
        
                >>> import torch
                >>> from efficientnet_pytorch_3d import EfficientNet3D
                >>> inputs = torch.rand(1, 3, 224, 224, 224)
                >>> model = EfficientNet3D.from_name('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        #bs = inputs.size(0)
        
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params, in_channels=in_channels)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet3D_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv3d = get_same_padding_conv3d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


# For use with segmentation/surrogate model
""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
class EfficientNet3DEncoder(EfficientNet3D):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels, weights):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[:self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):

            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.
                    x = module(x, drop_connect)

            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias")
        state_dict.pop("_fc.weight")
        super().load_state_dict(state_dict, **kwargs)





class EfficientAutoEncoder3D(EfficientNet3D):
    """EfficientNet AutoEncoder model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
        in_channels (int): Input data's channel number.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        
        
        >>> import torch
        >>> from efficientnet_pytorch_3d import EfficientNet3D
        >>> inputs = torch.rand(1, 3, 224, 224, 224)
        >>> model = EfficientNetAutoEncoder3D.from_name('efficientnet-b0')
        >>> model.eval()
        >>> ae_output, latent_fc_output = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None, in_channels=3):
        super().__init__(blocks_args=blocks_args, global_params=global_params, in_channels=in_channels)
        bn_mom = self._bn_mom
        bn_eps = self._bn_eps
        image_size = self._image_size
        block_args = self._last_block_args
        final_out_channels = in_channels
        
        Conv3d = get_same_padding_conv3d(image_size=image_size)
        self._feature_downsample = Conv3d(self._last_out_channels, 8, kernel_size=1, bias=False)
        self._downsample_bn = nn.BatchNorm3d(num_features=8, momentum=bn_mom, eps=bn_eps)
        self._feature_upsample = Conv3d(8, self._last_out_channels, kernel_size=1, bias=False)
        self._upsample_bn = nn.BatchNorm3d(num_features=self._last_out_channels, momentum=bn_mom, eps=bn_eps)
        self.feature_size = 8 * image_size[0]**2

        # EfficientNet Decoder
        # use dynamic image size for decoder
        TransposedConv3d = get_same_padding_conv3d(image_size=image_size, transposed=True)

        # Stem
        # self._decoder_conv_stem symmetry to self._conv_head
        in_channels = round_filters(1280, self._global_params)
        out_channels = block_args.output_filters  # output of final block
        self._decoder_conv_stem = TransposedConv3d(in_channels, out_channels, kernel_size=1, bias=False)
        image_size = calculate_output_image_size(image_size, 1, transposed=True)
        self._decoder_bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._decoder_blocks = nn.ModuleList([])
        assert len(self._blocks_image_size) == len(self._blocks_args) + 1
        self._blocks_image_size = list(reversed(self._blocks_image_size))
        for i, block_args in enumerate(reversed(self._blocks_args)):
            image_size = self._blocks_image_size[i]
            # Update block input and output filters based on depth multiplier.
            # input/output are flip here to support deconvolution
            block_args = block_args._replace(
                input_filters=round_filters(block_args.output_filters, self._global_params),
                output_filters=round_filters(block_args.input_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            # The first block needs to take care of stride and filter size increase.
            self._decoder_blocks.append(MBConvBlock3D(block_args, self._global_params, image_size=image_size,
                                                    decoder_mode=True, decoder_output_image_size=self._blocks_image_size[i+1]))
            image_size = self._blocks_image_size[i+1]
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._decoder_blocks.append(MBConvBlock3D(block_args, self._global_params, image_size=image_size,
                                                        decoder_mode=True, decoder_output_image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = round_filters(32, self._global_params)  # number of output channels
        out_channels = final_out_channels
        TransposedConv3d = get_same_padding_conv3d(image_size=global_params.image_size, transposed=True)
        self._decoder_conv_head = TransposedConv3d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._decoder_bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._sigmoid = torch.sigmoid

    def extract_features(self, inputs):
        """use convolution layer to extract feature,
        with additional down-sample layer to get 1280 hidden feature.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        x = super().extract_features(inputs)
        x = self._swish(self._downsample_bn(self._feature_downsample(x)))
        return x


    def decode_features(self, inputs):
        """decoder portion of this autoencoder.

        Args:
            inputs (tensor): Input tensor to the decoder, 
                             usually from self.extract_features

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # upsample
        x = self._swish(self._upsample_bn(self._feature_upsample(inputs)))
        
        # Stem
        x = self._swish(self._decoder_bn0(self._decoder_conv_stem(x)))
        
        # Blocks
        for idx, block in enumerate(self._decoder_blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        #x = self._swish(self._decoder_bn1(self._decoder_conv_head(x)))
        x = self._decoder_bn1(self._decoder_conv_head(x))
        return x
    
    def forward(self, inputs):
        """EfficientNet AutoEncoder's forward function.
        Calls extract_features to extract features, 
        then calls decode features to generates original inputs.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            (AE output tensor, latent representation tensor)
        """
        # Convolution layers
        x = self.extract_features(inputs)
        
        # Pooling and final linear layer
        latent_rep = x.flatten(start_dim=1)

        # Deconvolution - decoder
        x = self.decode_features(x)
        return x, latent_rep


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x


class EfficientUnet3D(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None**, **se** and **scse**.
            SE paper - https://arxiv.org/abs/1709.01507
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        freeze_encoder: Optional[bool] = False,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            freeze=freeze_encoder
        )

        self.decoder = EfficientUnet3DDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()


class EfficientUnetPlusPlus3D(SegmentationModel):
    """The EfficientUNet++ is a fully convolutional neural network for ordinary and medical image semantic segmentation. 
    Consists of an *encoder* and a *decoder*, connected by *skip connections*. The encoder extracts features of 
    different spatial resolutions, which are fed to the decoder through skip connections. The decoder combines its 
    own feature maps with the ones from skip connections to produce accurate segmentations masks.  The EfficientUNet++ 
    decoder architecture is based on the UNet++, a model composed of nested U-Net-like decoder sub-networks. To 
    increase performance and computational efficiency, the EfficientUNet++ replaces the UNet++'s blocks with 
    inverted residual blocks with depthwise convolutions and embedded spatial and channel attention mechanisms.
    Synergizes well with EfficientNet encoders. Due to their efficient visual representations (i.e., using few channels
    to represent extracted features), EfficientNet encoders require few computation from the decoder.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone) to extract features
        encoder_depth: Number of stages of the encoder, in range [3 ,5]. Each stage generate features two times smaller, 
            in spatial dimensions, than the previous one (e.g., for depth=0 features will haves shapes [(N, C, H, W)]), 
            for depth 1 features will have shapes [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in the decoder.
            Length of the list should be the same as **encoder_depth**
        in_channels: The number of input channels of the model, default is 3 (RGB images)
        classes: The number of classes of the output mask. Can be thought of as the number of channels of the mask
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is built 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **EfficientUnet++**

    Reference:
        https://arxiv.org/abs/2106.11447
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        squeeze_ratio: int = 1,
        expansion_ratio: int = 1,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        self.classes = classes
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = EfficientUnetPlusPlus3DDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            squeeze_ratio=squeeze_ratio,
            expansion_ratio=expansion_ratio
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "EfficientUNet++-{}".format(encoder_name)
        self.initialize()

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            output = self.forward(x)

        if self.classes > 1:
            probs = torch.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(x.size[1]),
                transforms.ToTensor()
            ]
        )
        full_mask = tf(probs.cpu())   

        return full_mask


encoders = {
    "efficientnet-b0": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b0"),
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (3, 5, 9, 16),
            "model_name": "efficientnet-b0",
        },
    },
    "efficientnet-b1": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b1"),
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (5, 8, 16, 23),
            "model_name": "efficientnet-b1",
        },
    },
    "efficientnet-b2": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b2"),
        "params": {
            "out_channels": (3, 32, 24, 48, 120, 352),
            "stage_idxs": (5, 8, 16, 23),
            "model_name": "efficientnet-b2",
        },
    },
    "efficientnet-b3": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b3"),
        "params": {
            "out_channels": (3, 40, 32, 48, 136, 384),
            "stage_idxs": (5, 8, 18, 26),
            "model_name": "efficientnet-b3",
        },
    },
    "efficientnet-b4": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b4"),
        "params": {
            "out_channels": (3, 48, 32, 56, 160, 448),
            "stage_idxs": (6, 10, 22, 32),
            "model_name": "efficientnet-b4",
        },
    },
    "efficientnet-b5": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b5"),
        "params": {
            "out_channels": (3, 48, 40, 64, 176, 512),
            "stage_idxs": (8, 13, 27, 39),
            "model_name": "efficientnet-b5",
        },
    },
    "efficientnet-b6": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b6"),
        "params": {
            "out_channels": (3, 56, 40, 72, 200, 576),
            "stage_idxs": (9, 15, 31, 45),
            "model_name": "efficientnet-b6",
        },
    },
    "efficientnet-b7": {
        "encoder": EfficientNet3DEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b7"),
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (11, 18, 38, 55),
            "model_name": "efficientnet-b7",
        },
    },
}

def get_encoder(name, in_channels=3, depth=5, weights=None, freeze=False):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None: # DAD changed to work with pytorch lightning autoencoder ckpts
        ckpt = torch.load(weights)
        state_dict = ckpt['state_dict']
        old_keys = list(state_dict.keys())
        decoder_keys = ('_feature', '_downsample', '_upsample', '_decoder')
        for old_key in old_keys:
            new_key = old_key[6:]
            state_dict[new_key] = state_dict.pop(old_key)
            if new_key.startswith(decoder_keys):
                state_dict.pop(new_key)
        
    encoder.set_in_channels(in_channels, weights)
    if weights is not None: encoder.load_state_dict(state_dict)

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder