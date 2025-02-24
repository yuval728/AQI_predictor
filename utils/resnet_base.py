import torch.nn as nn
from torchvision import models

class BaseEncoder(nn.Module):
    """
    A flexible encoder model that supports different architectures (ResNet, EfficientNet, etc.).
    """

    def __init__(self, arch="resnet18", pretrained=True, no_channels=3, dropout=0.5, add_block=False, num_frozen=0):
        super(BaseEncoder, self).__init__()

        self.add_block = add_block
        self.num_frozen = num_frozen

        # Dictionary of available architectures
        arch_dict = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b4": models.efficientnet_b4,
        }

        assert arch in arch_dict, f"Unsupported architecture: {arch}. Choose from {list(arch_dict.keys())}"

        # Load the model
        self.model = arch_dict[arch](weights="DEFAULT" if pretrained else None)

        if "resnet" in arch:
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()  
        elif "efficientnet" in arch:
            self.feature_dim = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity() 

        if no_channels != 3:
            if "resnet" in arch:
                self.model.conv1 = nn.Conv2d(no_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif "efficientnet" in arch:
                self.model.features[0][0] = nn.Conv2d(no_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


        if self.add_block:
            self.addition_block = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, self.feature_dim)
            )

        self.final_layers = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 512)
        )

        self.freeze_layers()

    def freeze_layers(self):
        """
        Freeze the first `num_frozen` layers of the model.
        """
        layers = list(self.model.children())
        assert 0 <= self.num_frozen <= len(layers), \
            f"num_frozen should be between 0 and {len(layers)}"

        for i, child in enumerate(layers):
            if i < self.num_frozen:
                for param in child.parameters():
                    param.requires_grad = False

        print(f"Number of frozen layers: {self.num_frozen}")

    def forward(self, x):
        x = self.model(x)
        if self.add_block:
            x = self.addition_block(x)
        x = self.final_layers(x)
        return x