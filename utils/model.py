import torch
import torch.nn as nn
from torchvision import models


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
            # "resnet101": models.resnet101,
            # EfficientNets
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            # "efficientnet_b3": models.efficientnet_b3,
            # "efficientnet_b4": models.efficientnet_b4,
            # "efficientnet_b5": models.efficientnet_b5,
            # "efficientnet_b6": models.efficientnet_b6,
            # "efficientnet_b7": models.efficientnet_b7,
            # MobileNets
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
            # Convnexts
            "convnext_tiny": models.convnext_tiny,
            "convnext_small": models.convnext_small,
            "convnext_base": models.convnext_base,
            # "convnext_large": models.convnext_large,
            # Vision Transformer
            "vit_base_16": models.vit_b_16,
            "vit_base_32": models.vit_b_32,
            # "vit_large_16": models.vit_l_16,
            # "vit_large_32": models.vit_l_32,
            # "vit_huge_14": models.vit_h_14,
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
            self.feature_dim = self.model.classifier[
                2
            ].in_features  # [LayerNorm, Flatten, Linear]
            self.model.classifier = self.model.classifier[:2]
        elif "vit" in arch:
            self.feature_dim = self.model.heads.head.in_features
            self.model.heads.head = nn.Identity()
        elif "swin" in arch:
            self.feature_dim = self.model.head.in_features
            self.model.head = nn.Identity()

        if no_channels != 3:
            if "resnet" in arch:
                self.model.conv1 = nn.Conv2d(
                    no_channels,
                    self.model.conv1.out_channels,
                    kernel_size=self.model.conv1.kernel_size,
                    stride=self.model.conv1.stride,
                    padding=self.model.conv1.padding,
                    bias=False,
                )

            elif "efficientnet" in arch:
                self.model.features[0][0] = nn.Conv2d(
                    no_channels,
                    self.model.features[0][0].out_channels,
                    kernel_size=self.model.features[0][0].kernel_size,
                    stride=self.model.features[0][0].stride,
                    padding=self.model.features[0][0].padding,
                    bias=False,
                )

            elif "mobilenet" in arch:
                self.model.features[0][0] = nn.Conv2d(
                    no_channels,
                    self.model.features[0][0].out_channels,
                    kernel_size=self.model.features[0][0].kernel_size,
                    stride=self.model.features[0][0].stride,
                    padding=self.model.features[0][0].padding,
                    bias=False,
                )

            elif "convnext" in arch:
                self.model.features[0][0] = nn.Conv2d(
                    no_channels,
                    self.model.features[0][0].out_channels,
                    kernel_size=self.model.features[0][0].kernel_size,
                    stride=self.model.features[0][0].stride,
                    padding=self.model.features[0][0].padding,
                    bias=self.model.features[0][0].bias is not None,
                )

            elif "vit" in arch:
                self.model.conv_proj = nn.Conv2d(
                    no_channels,
                    self.model.conv_proj.out_channels,
                    kernel_size=self.model.conv_proj.kernel_size,
                    stride=self.model.conv_proj.stride,
                    padding=self.model.conv_proj.padding,
                    bias=self.model.conv_proj.bias is not None,
                )

            elif "swin" in arch:
                first_conv_layer = self.model.features[0][
                    0
                ]  # Access the first Conv2d layer
                self.model.features[0][0] = nn.Conv2d(
                    no_channels,
                    first_conv_layer.out_channels,
                    kernel_size=first_conv_layer.kernel_size,
                    stride=first_conv_layer.stride,
                    padding=first_conv_layer.padding,
                    bias=first_conv_layer.bias is not None,
                )

        if self.add_block:
            self.addition_block = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, self.feature_dim),
            )

        self.freeze_layers()

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
        x = self.model(x)
        if self.add_block:
            x = self.addition_block(x)
        # x = self.final_layers(x)
        return x


class SigmoidGatedFusion2(nn.Module):
    """Independent Sigmoid-based gating for each input and then concatenates."""

    def __init__(self, in_dim1, in_dim2):
        super(SigmoidGatedFusion, self).__init__()
        self.attn1 = nn.Linear(in_dim1, in_dim1)  # Produces weighted features
        self.attn2 = nn.Linear(in_dim2, in_dim2)

    def forward(self, x1, x2):
        weight1 = torch.sigmoid(self.attn1(x1))
        weight2 = torch.sigmoid(self.attn2(x2))
        fused = torch.cat(
            (weight1 * x1, weight2 * x2), dim=1
        )  # Concatenate weighted features
        return fused


class SigmoidGatedFusion(nn.Module):
    """Independent Sigmoid-based gating for each input and then concatenates."""

    def __init__(self, in_dim1, in_dim2):
        super(SigmoidGatedFusion, self).__init__()
        self.attn1 = nn.Linear(in_dim1, 1)  # Produces weighted features
        self.attn2 = nn.Linear(in_dim2, 1)

    def forward(self, x1, x2):
        weight1 = torch.sigmoid(self.attn1(x1))
        weight2 = torch.sigmoid(self.attn2(x2))
        fused = torch.cat(
            (weight1 * x1, weight2 * x2), dim=1
        )  # Concatenate weighted features
        return fused


class SoftmaxGatedFusion(nn.Module):
    """Softmax-based gating where attention weights sum to 1."""

    def __init__(self, in_dim1, in_dim2):
        super(SoftmaxGatedFusion, self).__init__()
        self.gate = nn.Linear(in_dim1 + in_dim2, 2)  # Two attention scores

    def forward(self, x1, x2):
        combined_features = torch.cat((x1, x2), dim=1)
        weights = torch.softmax(
            self.gate(combined_features), dim=1
        )  # Softmax over two inputs
        weighted_x1 = weights[:, 0:1] * x1
        weighted_x2 = weights[:, 1:2] * x2
        fused = torch.cat(
            (weighted_x1, weighted_x2), dim=1
        )  # Concatenate weighted features
        return fused


class CrossAttention(nn.Module):
    """Multihead Cross Attention treating one input as query and other as key/value."""

    def __init__(self, in_dim1, in_dim2, shared_dim=256, num_heads=4):
        super(CrossAttention, self).__init__()

        # Project both inputs to a shared dimension
        self.query_proj = nn.Linear(in_dim1, shared_dim)
        self.key_proj = nn.Linear(in_dim2, shared_dim)
        self.value_proj = nn.Linear(in_dim2, shared_dim)

        # Multihead attention with consistent dimensions
        self.attn = nn.MultiheadAttention(
            embed_dim=shared_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x1, x2):
        # Project to shared dimension
        q = self.query_proj(x1).unsqueeze(1)  # (batch_size, 1, shared_dim)
        k = self.key_proj(x2).unsqueeze(1)  # (batch_size, 1, shared_dim)
        v = self.value_proj(x2).unsqueeze(1)  # (batch_size, 1, shared_dim)

        # Attention
        attn_output, _ = self.attn(q, k, v)
        attn_output = attn_output.squeeze(1)

        # Ensure consistent concatenation
        fused = torch.cat((attn_output, x1, x2), dim=1)

        # 🚀 Add the projection layer here
        projection = nn.Linear(fused.shape[1], 1024).to(fused.device)
        fused = projection(fused)

        return fused


def get_attention_module(attention_type, in_dim1, in_dim2, num_heads=4):
    """
    Factory method to return different attention mechanisms for concatenated features.

    Args:
    - attention_type (str): Type of attention ("sigmoid_gated", "softmax_gated", "cross")
    - in_dim1 (int): Feature dimension of input 1
    - in_dim2 (int): Feature dimension of input 2
    - num_heads (int): Number of heads (only for cross attention)

    Returns:
    - nn.Module: The selected attention mechanism
    """
    if attention_type == "sigmoid_gated":
        return SigmoidGatedFusion(in_dim1, in_dim2)
    elif attention_type == "softmax_gated":
        return SoftmaxGatedFusion(in_dim1, in_dim2)
    elif attention_type == "cross":
        return CrossAttention(in_dim1, in_dim2, num_heads)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


class AQIPrediction(nn.Module):
    """
    Unified model
    """

    def __init__(
        self,
        satellite_model,
        street_model,
        attention_type,
        dropout=0.5,
        num_classes=None,
    ):
        super(AQIPrediction, self).__init__()

        self.satellite_model = satellite_model
        self.street_model = street_model
        self.num_classes = num_classes
        self.attention_type = attention_type

        self.feature_dim = satellite_model.feature_dim + street_model.feature_dim

        if self.attention_type is not None:
            self.attention = get_attention_module(
                self.attention_type, satellite_model.feature_dim, street_model.feature_dim
            )

        self.final_layers = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if num_classes:
            self.final_layers[-1] = nn.Linear(128, num_classes)
        else:
            self.aqi_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())
            self.pm_head = nn.Sequential(nn.Linear(128, 2), nn.ReLU())  # PM2.5 & PM10
            self.gas_head = nn.Sequential(
                nn.Linear(128, 4), nn.ReLU()
            )  # O3, CO, SO2, NO2

    def forward(self, street_img, satellite_img):
        street_features = self.street_model(street_img)
        satellite_features = self.satellite_model(satellite_img)
        # print(street_features.shape, satellite_features.shape)
        
        if self.attention_type is None:
            features = torch.cat((street_features, satellite_features), dim=1)
        else:
            features = self.attention(satellite_features, street_features)

        output = self.final_layers(features)
        if self.num_classes:
            return output

        return self.aqi_head(output), self.pm_head(output), self.gas_head(output)
