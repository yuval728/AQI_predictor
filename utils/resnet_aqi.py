import torch
import torch.nn as nn



class AQIPrediction(nn.Module):
    """
    Unified model
    """
    
    def __init__(self, satellite_model, street_model, dropout=0.5, num_classes=None):
        
        super(AQIPrediction, self).__init__()

        layer1=512

        self.street_model = street_model
        
        if satellite_model:
            self.satellite_model = satellite_model
            layer1=512*2

        self.final_layers = nn.Sequential(
            nn.Linear(layer1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        if num_classes:
            self.final_layers[-1] = nn.Linear(128, num_classes)
            
    def forward(self, street_img, satellite_img=None):
            
        
        street_features = self.street_model(street_img)
        if satellite_img is not None:
            if self.more_bands: # Convert to tensor from npy
                satellite_img = torch.tensor(satellite_img).float()
            satellite_features = self.satellite_model(satellite_img)
            features = torch.cat((street_features, satellite_features), dim=1)
            output = self.final_layers(features)
        else:
            output = self.final_layers(street_features)
        return output