import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Architecture CNN légère optimisée pour les patchs 96x96.
    Entrée : (Batch, 3, 96, 96)
    Sortie : (Batch, 1) - Probabilité de tumeur
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Bloc 1 : Extraction bas niveau
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # -> 48x48
        
        # Bloc 2 : Features intermédiaires
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # -> 24x24
        
        # Bloc 3 : Features haut niveau
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) # -> 12x12
        
        # Classification
        self.flatten_dim = 128 * 12 * 12
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Feature Extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flattening
        x = x.view(-1, self.flatten_dim)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Note: On retourne les logits (pas de Sigmoid ici si on utilise BCEWithLogitsLoss)
        return x
