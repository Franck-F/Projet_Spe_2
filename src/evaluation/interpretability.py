"""
Interprétabilité des modèles avec Grad-CAM
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class GradCAM:
    """
    Grad-CAM pour visualiser les régions importantes
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Modèle PyTorch
            target_layer: Couche cible pour Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Enregistrer les hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        """Hook pour sauvegarder les activations"""
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook pour sauvegarder les gradients"""
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_image: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Génère la heatmap Grad-CAM
        
        Args:
            input_image: Image d'entrée (1, C, H, W)
            target_class: Classe cible (None pour la classe prédite)
            
        Returns:
            Heatmap Grad-CAM
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Calculer les poids
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Calculer la CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normaliser
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def visualize_gradcam(image: np.ndarray,
                      cam: np.ndarray,
                      alpha: float = 0.5) -> go.Figure:
    """
    Visualise l'image avec la heatmap Grad-CAM
    
    Args:
        image: Image originale (H, W, 3)
        cam: Heatmap Grad-CAM (H, W)
        alpha: Transparence de la heatmap
        
    Returns:
        Figure Plotly
    """
    # Redimensionner la CAM à la taille de l'image
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Appliquer une colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superposer
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    
    # Créer la figure avec subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Image Originale', 'Heatmap Grad-CAM', 'Superposition')
    )
    
    # Image originale
    fig.add_trace(
        go.Image(z=image),
        row=1, col=1
    )
    
    # Heatmap
    fig.add_trace(
        go.Image(z=heatmap),
        row=1, col=2
    )
    
    # Superposition
    fig.add_trace(
        go.Image(z=overlay),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Visualisation Grad-CAM',
        showlegend=False,
        width=1200,
        height=400
    )
    
    return fig


def analyze_prediction_errors(model: nn.Module,
                              dataloader,
                              device: str = 'cuda',
                              num_examples: int = 10):
    """
    Analyse les erreurs de prédiction
    
    Args:
        model: Modèle PyTorch
        dataloader: DataLoader
        device: Device
        num_examples: Nombre d'exemples à analyser
        
    Returns:
        Liste de (image, true_label, pred_label, confidence)
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # Trouver les erreurs
            mask = preds != labels
            
            for i in range(inputs.size(0)):
                if mask[i] and len(errors) < num_examples:
                    errors.append({
                        'image': inputs[i].cpu().numpy(),
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'confidence': probs[i].max().item()
                    })
            
            if len(errors) >= num_examples:
                break
    
    return errors
