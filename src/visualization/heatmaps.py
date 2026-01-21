"""
Visualisation des heatmaps et Grad-CAM
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cv2


def plot_gradcam_heatmap(image: np.ndarray,
                         heatmap: np.ndarray,
                         title: str = 'Grad-CAM') -> go.Figure:
    """
    Visualise une heatmap Grad-CAM
    
    Args:
        image: Image originale (H, W, 3)
        heatmap: Heatmap (H, W)
        title: Titre
        
    Returns:
        Figure Plotly
    """
    # Redimensionner la heatmap
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Appliquer colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superposer
    overlay = (0.6 * heatmap_colored + 0.4 * image).astype(np.uint8)
    
    # Créer la figure
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Image', 'Heatmap', 'Superposition')
    )
    
    fig.add_trace(go.Image(z=image), row=1, col=1)
    fig.add_trace(go.Image(z=heatmap_colored), row=1, col=2)
    fig.add_trace(go.Image(z=overlay), row=1, col=3)
    
    fig.update_layout(
        title=title,
        showlegend=False,
        width=1200,
        height=400
    )
    
    return fig


def plot_attention_map(image: np.ndarray,
                       attention: np.ndarray) -> go.Figure:
    """
    Visualise une carte d'attention
    
    Args:
        image: Image originale
        attention: Carte d'attention
        
    Returns:
        Figure Plotly
    """
    # Normaliser l'attention
    attention_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    # Redimensionner
    attention_resized = cv2.resize(attention_norm, (image.shape[1], image.shape[0]))
    
    # Créer la figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Image Originale', 'Carte d\'Attention')
    )
    
    fig.add_trace(go.Image(z=image), row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            z=attention_resized,
            colorscale='Viridis',
            showscale=True
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Carte d\'Attention',
        showlegend=False,
        width=1000,
        height=500
    )
    
    return fig


def plot_multiple_heatmaps(images: list,
                           heatmaps: list,
                           titles: list = None,
                           n_cols: int = 3) -> go.Figure:
    """
    Affiche plusieurs heatmaps en grille
    
    Args:
        images: Liste d'images
        heatmaps: Liste de heatmaps
        titles: Liste de titres
        n_cols: Nombre de colonnes
        
    Returns:
        Figure Plotly
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    if titles is None:
        titles = [f'Exemple {i+1}' for i in range(n_images)]
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles
    )
    
    for idx, (img, heatmap) in enumerate(zip(images, heatmaps)):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Superposer heatmap et image
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (0.6 * heatmap_colored + 0.4 * img).astype(np.uint8)
        
        fig.add_trace(
            go.Image(z=overlay),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Visualisations Grad-CAM',
        showlegend=False,
        height=400 * n_rows,
        width=400 * n_cols
    )
    
    return fig
