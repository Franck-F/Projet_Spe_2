"""
CNN Baseline pour classification de patchs
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    CNN simple from scratch pour établir une baseline
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        """
        Args:
            num_classes: Nombre de classes (2: normal/tumoral)
            dropout: Taux de dropout
        """
        super(BaselineCNN, self).__init__()
        
        # Couches convolutionnelles
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Couches fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 14 * 14, 512),  # Pour input 224x224
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)):
    """
    Affiche un résumé du modèle
    
    Args:
        model: Modèle PyTorch
        input_size: Taille de l'input
    """
    from torchsummary import summary
    summary(model, input_size)
