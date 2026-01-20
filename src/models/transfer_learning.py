"""
Transfer Learning avec modèles pré-entraînés
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


def get_pretrained_model(
    architecture: Literal['resnet50', 'resnet101', 'efficientnet-b3', 'densenet121'],
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Charge un modèle pré-entraîné et adapte la dernière couche
    
    Args:
        architecture: Architecture à utiliser
        num_classes: Nombre de classes de sortie
        pretrained: Utiliser les poids ImageNet
        freeze_backbone: Geler les couches du backbone
        
    Returns:
        Modèle PyTorch
    """
    if architecture == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            # Dégeler seulement la dernière couche
            for param in model.fc.parameters():
                param.requires_grad = True
                
    elif architecture == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
                
    elif architecture == 'efficientnet-b3':
        model = models.efficientnet_b3(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
                
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Architecture non supportée: {architecture}")
    
    return model


def unfreeze_layers(model: nn.Module, num_layers: int = -1):
    """
    Dégèle progressivement les couches du modèle
    
    Args:
        model: Modèle PyTorch
        num_layers: Nombre de couches à dégeler (-1 pour toutes)
    """
    if num_layers == -1:
        # Dégeler toutes les couches
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Dégeler les num_layers dernières couches
        # TODO: Implémenter dégelage progressif
        pass
