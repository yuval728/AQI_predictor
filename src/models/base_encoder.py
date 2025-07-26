"""
Base encoder architectures for the AQI prediction model.
Supports various CNN architectures with configurable input channels and layers.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BaseEncoder(nn.Module):
    """
    A flexible encoder model that supports different architectures (ResNet, EfficientNet, etc.).
    """

    def __init__(
        self,
        arch="resnet18",
        pretrained=True,
        no_channels=3,
        dropout=0.5,
        add_block=False,
        num_frozen=0,
    ):
        super(BaseEncoder, self).__init__()

        self.add_block = add_block
        self.num_frozen = num_frozen

        # Dictionary of available architectures
        arch_dict = {
            # ResNets
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            # EfficientNets
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            # MobileNets
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
            # Convnexts
            "convnext_tiny": models.convnext_tiny,
            "convnext_small": models.convnext_small,
            "convnext_base": models.convnext_base,
            # Vision Transformer
            "vit_base_16": models.vit_b_16,
            "vit_base_32": models.vit_b_32,
            # Swin Transformer V2
            "swinv2_tiny": models.swin_v2_t,
            "swinv2_small": models.swin_v2_s,
            "swinv2_base": models.swin_v2_b,
        }

        assert arch in arch_dict, (
            f"Unsupported architecture: {arch}. Choose from {list(arch_dict.keys())}"
        )

        # Load the model
        self.model = arch_dict[arch](weights="DEFAULT" if pretrained else None)

        # Modify architecture-specific layers
        if "resnet" in arch:
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif "efficientnet" in arch:
            self.feature_dim = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
        elif "mobilenet" in arch:
            self.feature_dim = self.model.classifier[0].in_features
            self.model.classifier = nn.Identity()
        elif "convnext" in arch:
            self.feature_dim = self.model.classifier[2].in_features
            self.model.classifier = self.model.classifier[:2]
        elif "vit" in arch:
            self.feature_dim = self.model.heads.head.in_features
            self.model.heads.head = nn.Identity()
        elif "swin" in arch:
            self.feature_dim = self.model.head.in_features
            self.model.head = nn.Identity()

        # Modify input layer for custom number of channels
        if no_channels != 3:
            self._modify_input_layer(no_channels)

        # Add an optional additional block
        if self.add_block:
            self.addition_block = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, self.feature_dim),
            )

    def _modify_input_layer(self, no_channels):
        """Modify the input layer to accept a different number of channels."""
        if "resnet" in self.model.__class__.__name__.lower():
            self.model.conv1 = nn.Conv2d(
                no_channels,
                self.model.conv1.out_channels,
                kernel_size=self.model.conv1.kernel_size,
                stride=self.model.conv1.stride,
                padding=self.model.conv1.padding,
                bias=False,
            )
        elif "efficientnet" in self.model.__class__.__name__.lower():
            self.model.features[0][0] = nn.Conv2d(
                no_channels,
                self.model.features[0][0].out_channels,
                kernel_size=self.model.features[0][0].kernel_size,
                stride=self.model.features[0][0].stride,
                padding=self.model.features[0][0].padding,
                bias=False,
            )
        elif "mobilenet" in self.model.__class__.__name__.lower():
            self.model.features[0][0] = nn.Conv2d(
                no_channels,
                self.model.features[0][0].out_channels,
                kernel_size=self.model.features[0][0].kernel_size,
                stride=self.model.features[0][0].stride,
                padding=self.model.features[0][0].padding,
                bias=False,
            )
        elif "convnext" in self.model.__class__.__name__.lower():
            self.model.features[0][0] = nn.Conv2d(
                no_channels,
                self.model.features[0][0].out_channels,
                kernel_size=self.model.features[0][0].kernel_size,
                stride=self.model.features[0][0].stride,
                padding=self.model.features[0][0].padding,
                bias=self.model.features[0][0].bias is not None,
            )
        elif "vit" in self.model.__class__.__name__.lower():
            self.model.conv_proj = nn.Conv2d(
                no_channels,
                self.model.conv_proj.out_channels,
                kernel_size=self.model.conv_proj.kernel_size,
                stride=self.model.conv_proj.stride,
                padding=self.model.conv_proj.padding,
                bias=self.model.conv_proj.bias is not None,
            )
        elif "swin" in self.model.__class__.__name__.lower():
            first_conv_layer = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                no_channels,
                first_conv_layer.out_channels,
                kernel_size=first_conv_layer.kernel_size,
                stride=first_conv_layer.stride,
                padding=first_conv_layer.padding,
                bias=first_conv_layer.bias is not None,
            )

    def freeze_layers(self):
        """
        Freeze the first `num_frozen` layers of the model.
        """
        layers = list(self.model.children())
        assert 0 <= self.num_frozen <= len(layers), (
            f"num_frozen should be between 0 and {len(layers)}"
        )

        for i, child in enumerate(layers):
            if i < self.num_frozen:
                for param in child.parameters():
                    param.requires_grad = False

        print(f"Number of frozen layers: {self.num_frozen}")

    def forward(self, x):
        """
        Forward pass through the encoder.
        """
        x = self.model(x)
        if self.add_block:
            x = self.addition_block(x)
        return x


# Utility functions for creating common encoder configurations
def create_satellite_encoder(
    arch: str = "resnet18",
    channels: int = 7,
    add_block: bool = False,
    num_frozen: int = 0,
) -> BaseEncoder:
    """Create an encoder optimized for satellite imagery."""
    return BaseEncoder(
        arch=arch,
        no_channels=channels,
        add_block=add_block,
        num_frozen=num_frozen,
    )


def create_street_encoder(
    arch: str = "resnet18",
    channels: int = 3,
    add_block: bool = False,
    num_frozen: int = 0,
) -> BaseEncoder:
    """Create an encoder optimized for street view imagery."""
    return BaseEncoder(
        arch=arch,
        no_channels=channels,
        add_block=add_block,
        num_frozen=num_frozen,
    )


# Available architectures and their feature dimensions
ARCHITECTURE_INFO = {
    "resnet18": {"default_features": 512, "pretrained": True},
    "resnet34": {"default_features": 512, "pretrained": True},
    "resnet50": {"default_features": 2048, "pretrained": True},
    "efficientnet_b0": {"default_features": 1280, "pretrained": True},
    "mobilenet_v2": {"default_features": 1280, "pretrained": True},
}
