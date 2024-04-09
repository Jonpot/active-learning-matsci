import torch
import torch.nn as nn
from torch.nn import functional as F
import math


try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


# Swish activation function
if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class PreActivatedConv3dReLU(nn.Sequential):
    """
    Pre-activated 2D convolution, as proposed in https://arxiv.org/pdf/1603.05027.pdf. Feature maps are processed by a normalization layer, 
    followed by a ReLU activation and a 3x3 convolution.
    normalization
    """
    def __init__(
            self,
            in_channels, 
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )
        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()
        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()

        relu = nn.ReLU(inplace=True)

        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        super(PreActivatedConv3dReLU, self).__init__(conv, bn, relu)


class Conv3dReLU(nn.Sequential):
    """
    Block composed of a 3x3 convolution followed by a normalization layer and ReLU activation.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()
        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm3d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DepthWiseConv3d(nn.Conv3d):
    "Depth-wise convolution operation"
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__(channels, channels, kernel_size, stride=stride, padding=kernel_size//2, groups=channels)


class PointWiseConv3d(nn.Conv3d):
    "Point-wise (1x1) convolution operation"
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1)


class Conv3dDynamicSamePadding(nn.Conv3d):
    """3D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height or depth
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv3d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3

    def forward(self, x):
        ih, iw, iz = x.size()[-3:]
        kh, kw, kz = self.weight.size()[-3:]
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.stride[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2,
                          pad_z // 2, pad_z - pad_z // 2])
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv3dStaticSamePadding(nn.Conv3d):
    """3D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv3dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw, iz = (image_size, image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw, kz = self.weight.size()[-3:]
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.stride[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            self.static_padding = nn.ZeroPad3d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2,
                                                pad_z // 2, pad_z - pad_z // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class TransposedConv3dStaticSamePadding(nn.ConvTranspose3d):
    """3D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #         op: output padding
    #     Output after ConvTranspose3d:
    #         (i-1)*s + (k-1)*d + op + 1

    def __init__(self, in_channels, out_channels, kernel_size, image_size, stride=1, output_padding=0, groups=1, bias=True, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, output_padding, groups, bias, dilation)
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3
        self.output_padding = output_padding
        # NOTE: image_size here represents the desired output image_size
        oh, ow, oz = (image_size, image_size, image_size) if isinstance(image_size, int) else image_size
        self._oh, self._ow, self._oz = oh, ow, oz
        sh, sw, sz = self.stride
        ih, iw, iz = math.ceil(oh / sh), math.ceil(ow / sw), math.ceil(oz / sz)
        self._ih, self._iw, self._iz = ih, iw, iz
        kh, kw, kz = self.weight.size()[-3:]
        
        # actual height/width after TransposedConv3d
        actual_oh = (ih - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + self.output_padding + 1
        actual_ow = (iw - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + self.output_padding + 1
        actual_oz = (iz - 1) * self.stride[2] + (kz - 1) * self.dilation[2] + self.output_padding + 1
        
        crop_h = actual_oh - oh
        crop_w = actual_ow - ow
        crop_z = actual_oz - oz
        
        assert crop_h >= 0 and crop_w >= 0 and crop_z >= 0
        self._crop_h = crop_h
        self._crop_w = crop_w
        self._crop_z = crop_z
        
        self._actual_oh = actual_oh
        self._actual_ow = actual_ow
        self._actual_oz = actual_oz

    def forward(self, x):
        # assert x.size()[-3:] == (self._ih, self._iw, self._iz)
        x = F.conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)
        # assert x.size()[-3:] == (self._actual_oh,  self._actual_ow,  self._actual_oz)
        crop_h, crop_w, crop_z = self._crop_h, self._crop_w, self._crop_z
        if crop_h > 0 or crop_w > 0 or crop_z > 0:
            x = x[:, :, crop_h // 2 : - (crop_h - crop_h // 2),
                        crop_w // 2 : - (crop_w - crop_w // 2),
                        crop_z // 2 : - (crop_z - crop_z // 2)]
        # assert x.size()[-3:] == (self._oh, self._ow, self._oz)
        return x


class MaxPool3dDynamicSamePadding(nn.MaxPool2d):
    """3D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * 3 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 3 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 3 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x):
        ih, iw, iz = x.size()[-3:]
        kh, kw, kz = self.kernel_size
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.stride[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2,
                          pad_z // 2, pad_z - pad_z // 2])
        return F.max_pool3d(x, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)


class MaxPool3dStaticSamePadding(nn.MaxPool2d):
    """3D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 3 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 3 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 3 if isinstance(self.dilation, int) else self.dilation

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw, iz = (image_size, image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw, kz = self.kernel_size
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.stride[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            self.static_padding = nn.ZeroPad3d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2,
                                                pad_z // 2, pad_z - pad_z // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool3d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class SEModule(nn.Module):
    """
    Spatial squeeze & channel excitation attention module, as proposed in https://arxiv.org/abs/1709.01507.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x)


class sSEModule(nn.Module):
    """
    Channel squeeze & spatial excitation attention module, as proposed in https://arxiv.org/abs/1808.08127.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.sSE(x)


class SCSEModule(nn.Module):
    """
    Concurrent spatial and channel squeeze & excitation attention module, as proposed in https://arxiv.org/pdf/1803.02579.pdf.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv3d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        elif name == 'se':
            self.attention = SEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
