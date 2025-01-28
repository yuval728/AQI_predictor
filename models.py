import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseResnet18(nn.Module):
    """
    Base encoder model
    """
    
    def __init__(self, no_channels=3, dropout=0.5, add_block=False, num_frozen=0):
        
        super(BaseResnet18, self).__init__()

        self.add_block = add_block
        self.num_frozen = num_frozen

        self.model= models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if no_channels != 3:
            self.model.conv1 = nn.Conv2d(no_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if self.add_block:
            self.addition_block = nn.Sequential(
                nn.Linear(in_features=1000, out_features=1000),
                nn.BatchNorm1d(1000),
                # nn.LayerNorm(1000),
                nn.Dropout(dropout),
                nn.Linear(in_features=1000, out_features=1000)
            )
    
        self.final_layers = nn.Sequential(
            nn.Linear(in_features=1000, out_features=512),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=512)
        )
        
        self.freeze_layers()
        
        
    def freeze_layers(self):
        """
        Freeze layers of the model
        """
        
        assert (62 >= self.num_frozen >= 0), "Number of frozen layers should be between 0 and 62"
        counter = 0
        for i, param in enumerate(self.model.parameters()):
            if i < self.num_frozen:
                param.requires_grad = False
            counter += 1
        print(f"Number of frozen layers: {counter}")
        
    def forward(self, x):
        
        x = self.model(x)
        print(x.shape)
        if self.add_block:
            x = self.addition_block(x)
        print(x.shape)
        x = self.final_layers(x)
        
        return x


class ResnetRegression(nn.Module):
    """
    Regression model
    """
    
    def __init__(self, no_channels=3, dropout=0.5, add_block=False, num_frozen=0):
        
        super(ResnetRegression, self).__init__()
        
        self.encoder = BaseResnet18(no_channels=no_channels, dropout=dropout, add_block=add_block, num_frozen=num_frozen)
        self.encoder.final_layers[3] = nn.Linear(in_features=512, out_features=1)
        
    def forward(self, x):
    
        x = self.encoder(x)
        return x
    

class ResnetClassification(nn.Module):
    """
    Classification model
    """
    
    def __init__(self, no_channels=3, num_classes=3, dropout=0.5, add_block=False, num_frozen=0):
        
        super(ResnetClassification, self).__init__()
        
        self.encoder = BaseResnet18(no_channels=no_channels, dropout=dropout, add_block=add_block, num_frozen=num_frozen)
        self.encoder.final_layers[3] = nn.Linear(in_features=512, out_features=num_classes)
        
    def forward(self, x):
    
        x = self.encoder(x)
        return x
         
        
def loss_fn_regression(outputs, targets):
    """
    Loss function for regression
    """
    
    return nn.MSELoss()(outputs, targets)

def loss_fn_classification(outputs, targets):
    """
    Loss function for classification
    """
    
    return nn.CrossEntropyLoss()(outputs, targets)

def accuracy(outputs, targets):
    """
    Accuracy function
    """
    
    return (outputs.argmax(1) == targets).float().mean()

def rmse(outputs, targets):
    """
    RMSE function
    """
    
    return torch.sqrt(nn.MSELoss()(outputs, targets))

def mae(outputs, targets):
    """
    MAE function
    """
    
    return nn.L1Loss()(outputs, targets)


if __name__ == "__main__":
    
    model = ResnetRegression(add_block=True)
    # print(model)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3)
    outputs = model(x)
    print(outputs.shape)
    
    
    model = ResnetClassification()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3)
    
    outputs = model(x)
    print(outputs.shape)
    