from .__version__ import __version__
from .models import EfficientNet3D, EfficientAutoEncoder3D, EfficientUnet3D, EfficientUnetPlusPlus3D


'''
def create_model(
    arch: str,
    encoder_name: str = "efficientnet-b0",
    encoder_weights: Optional[str] = None,
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parameters

    """
    archs = [EfficientNet3D, EfficientAutoEncoder3D, EfficientUnet3D, EfficientUnetPlusPlus3D]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(f"Wrong architecture type `{arch}`. Avalibale options are: {list(archs_dict.keys()}")
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
'''
