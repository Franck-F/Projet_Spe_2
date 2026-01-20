"""
Visualisations Plotly pour l'EDA
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_class_distribution(df: pd.DataFrame, 
                            class_column: str = 'label',
                            title: str = 'Distribution des Classes') -> go.Figure:
    """
    Visualise la distribution des classes
    
    Args:
        df: DataFrame avec les données
        class_column: Nom de la colonne des classes
        title: Titre du graphique
        
    Returns:
        Figure Plotly
    """
    counts = df[class_column].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts.index,
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker_color=['#3498db', '#e74c3c']
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Classe',
        yaxis_title='Nombre de Patchs',
        showlegend=False,
        width=700,
        height=500
    )
    
    return fig


def plot_hospital_distribution(df: pd.DataFrame,
                               hospital_column: str = 'hospital',
                               class_column: str = 'label') -> go.Figure:
    """
    Visualise la distribution par hôpital
    
    Args:
        df: DataFrame avec les données
        hospital_column: Nom de la colonne hôpital
        class_column: Nom de la colonne classe
        
    Returns:
        Figure Plotly
    """
    grouped = df.groupby([hospital_column, class_column]).size().reset_index(name='count')
    
    fig = px.bar(
        grouped,
        x=hospital_column,
        y='count',
        color=class_column,
        title='Distribution des Classes par Hôpital',
        labels={'count': 'Nombre de Patchs', hospital_column: 'Hôpital'},
        barmode='group',
        color_discrete_map={0: '#3498db', 1: '#e74c3c'}
    )
    
    fig.update_layout(
        width=900,
        height=500
    )
    
    return fig


def plot_patient_stage_distribution(df: pd.DataFrame,
                                    stage_column: str = 'pn_stage') -> go.Figure:
    """
    Visualise la distribution des stades pN
    
    Args:
        df: DataFrame avec les données patients
        stage_column: Nom de la colonne stade pN
        
    Returns:
        Figure Plotly
    """
    counts = df[stage_column].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f'pN{i}' for i in counts.index],
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker_color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        )
    ])
    
    fig.update_layout(
        title='Distribution des Stades pN',
        xaxis_title='Stade pN',
        yaxis_title='Nombre de Patients',
        showlegend=False,
        width=700,
        height=500
    )
    
    return fig


def plot_patch_samples(images: list,
                       labels: list,
                       predictions: list = None,
                       n_cols: int = 5) -> go.Figure:
    """
    Affiche une grille d'exemples de patchs
    
    Args:
        images: Liste d'images
        labels: Liste de labels
        predictions: Liste de prédictions (optionnel)
        n_cols: Nombre de colonnes
        
    Returns:
        Figure Plotly
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f'Label: {l}' + (f'\nPred: {p}' if predictions else '') 
                       for l, p in zip(labels, predictions or [None]*n_images)]
    )
    
    for idx, img in enumerate(images):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Image(z=img),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Exemples de Patchs',
        showlegend=False,
        height=300 * n_rows,
        width=250 * n_cols
    )
    
    return fig


def plot_training_history(train_losses: list,
                          val_losses: list,
                          train_accs: list = None,
                          val_accs: list = None) -> go.Figure:
    """
    Visualise l'historique d'entraînement
    
    Args:
        train_losses: Losses d'entraînement
        val_losses: Losses de validation
        train_accs: Accuracies d'entraînement (optionnel)
        val_accs: Accuracies de validation (optionnel)
        
    Returns:
        Figure Plotly
    """
    epochs = list(range(1, len(train_losses) + 1))
    
    # Créer subplots si accuracies fournies
    if train_accs and val_accs:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss', 'Accuracy')
        )
        
        # Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=train_losses, name='Train Loss', mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_losses, name='Val Loss', mode='lines+markers'),
            row=1, col=1
        )
        
        # Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=train_accs, name='Train Acc', mode='lines+markers'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=val_accs, name='Val Acc', mode='lines+markers'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        fig.update_yaxes(title_text='Loss', row=1, col=1)
        fig.update_yaxes(title_text='Accuracy (%)', row=1, col=2)
        
        fig.update_layout(
            title='Historique d\'Entraînement',
            width=1200,
            height=500
        )
    else:
        # Seulement loss
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs, y=train_losses,
            name='Train Loss',
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs, y=val_losses,
            name='Val Loss',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Historique d\'Entraînement - Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            width=800,
            height=500
        )
    
    return fig
