"""
AQI Prediction Model with Multi-modal Learning and Attention Mechanisms.
"""

import torch.nn as nn
from .attention import get_attention_module


class AQIPrediction(nn.Module):
    """
    Unified model for AQI prediction with multi-modal inputs and multi-task outputs.
    """
    
    def __init__(self, satellite_model, street_model, attention_type="sigmoid_gated", dropout=0.5, num_classes=None):
        
        super(AQIPrediction, self).__init__()
        
        self.satellite_model = satellite_model
        self.street_model = street_model
        self.num_classes = num_classes
        
        self.feature_dim = satellite_model.feature_dim + street_model.feature_dim
        
        self.attention = get_attention_module(attention_type, satellite_model.feature_dim, street_model.feature_dim)
        
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
            self.classifier = nn.Linear(128, num_classes)
        else:
            # Multi-task regression heads
            # self.aqi_head = nn.Sequential(nn.Linear(128, 1), nn.ReLU())  
            # self.pm_head = nn.Sequential(nn.Linear(128, 2), nn.ReLU())   # PM2.5 & PM10
            # self.gas_head = nn.Sequential(nn.Linear(128, 4), nn.ReLU())  # O3, CO, SO2, NO2
            
            self.aqi_head = nn.Linear(128, 1)  # Single output for AQI
            self.pm_head = nn.Linear(128, 2)   # PM2.5 & PM10
            self.gas_head = nn.Linear(128, 4)  # O3, CO, SO2, NO2
            
    def forward(self, street_img, satellite_img):
        
        street_features = self.street_model(street_img)
        satellite_features = self.satellite_model(satellite_img)
        # print(street_features.shape, satellite_features.shape)
        
        # Apply attention mechanism
        features = self.attention(satellite_features, street_features)
        
        output = self.final_layers(features)
        
        if self.num_classes:
            return self.classifier(output)
            
        return self.aqi_head(output), self.pm_head(output), self.gas_head(output)


def create_aqi_model(satellite_encoder, street_encoder, attention_type="sigmoid_gated", dropout=0.5, num_classes=None):
    """
    Factory function to create AQI prediction model.
    """
    return AQIPrediction(
        satellite_model=satellite_encoder,
        street_model=street_encoder,
        attention_type=attention_type,
        dropout=dropout,
        num_classes=num_classes
    )
