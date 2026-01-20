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
        
        
        # Calcul automatique de la taille de l'input pour la couche dense
        self._to_linear = None
        self._get_conv_output((3, 96, 96)) # Initialisation pour 96x96 par défaut, mais s'adapte
        
        # Couches fully connected
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._to_linear, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.features(input)
        self._to_linear = int(output_feat.data.view(batch_size, -1).size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        # Recalculer la taille si l'input change drastiquement (optionnel, ici on suppose fixe après init)
        # Pour être robuste, on peut recalculer self._to_linear si les dimensions changent, 
        # mais on ne peut pas changer la couche Linear dynamiquement en training.
        # Donc on suppose que l'input size est cohérent avec l'init.
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model_summary(model: nn.Module, input_size: tuple = (3, 96, 96)):
    """
    Affiche un résumé du modèle
    
    Args:
        model: Modèle PyTorch
        input_size: Taille de l'input (C, H, W)
    """
    from torchsummary import summary
    summary(model, input_size)
