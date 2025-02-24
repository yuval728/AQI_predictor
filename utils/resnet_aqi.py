import torch
import torch.nn as nn



class AQIPrediction(nn.Module):
    """
    Unified model
    """
    
    def __init__(self, satellite_model, street_model, dropout=0.5, num_classes=None):
        
        super(AQIPrediction, self).__init__()
        
        self.satellite_model = satellite_model
        self.street_model = street_model
        
        self.final_layers = nn.Sequential(
            nn.Linear(512 + 512, 512),
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
            
    def forward(self, street_img, satellite_img):
        
        street_features = self.street_model(street_img)
        satellite_features = self.satellite_model(satellite_img)
        
        features = torch.cat((street_features, satellite_features), dim=1)
        output = self.final_layers(features)
        
        return output