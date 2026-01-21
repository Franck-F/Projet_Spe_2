"""
Métriques d'évaluation pour la classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    cohen_kappa_score
)
from typing import Dict
import plotly.figure_factory as ff
import plotly.graph_objects as go


def compute_all_metrics(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Calcule toutes les métriques de classification
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        y_proba: Probabilités prédites (optionnel, pour AUC)
        
    Returns:
        Dictionnaire de métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Métriques nécessitant les probabilités
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            metrics['auc_pr'] = average_precision_score(y_true, y_proba, average='weighted')
        except:
            pass
    
    # Cohen's Kappa (pour accord)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    return metrics


def compute_f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcule le F2-score (pondération vers recall)
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        
    Returns:
        F2-score
    """
    from sklearn.metrics import fbeta_score
    return fbeta_score(y_true, y_pred, beta=2, average='weighted')


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list = None) -> go.Figure:
    """
    Crée une matrice de confusion avec Plotly
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        class_names: Noms des classes
        
    Returns:
        Figure Plotly
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    # Normaliser par ligne (rappel par classe)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Créer la heatmap
    fig = ff.create_annotated_heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        annotation_text=cm,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Matrice de Confusion',
        xaxis_title='Prédiction',
        yaxis_title='Vérité Terrain',
        width=600,
        height=600
    )
    
    return fig


def plot_roc_curve(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   class_names: list = None) -> go.Figure:
    """
    Trace la courbe ROC
    
    Args:
        y_true: Labels réels
        y_proba: Probabilités prédites
        class_names: Noms des classes
        
    Returns:
        Figure Plotly
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binariser les labels
    n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig = go.Figure()
    
    # Courbe ROC pour chaque classe
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        class_name = class_names[i] if class_names else f'Class {i}'
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{class_name} (AUC = {roc_auc:.3f})'
        ))
    
    # Ligne diagonale
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))
    
    fig.update_layout(
        title='Courbe ROC',
        xaxis_title='Taux de Faux Positifs',
        yaxis_title='Taux de Vrais Positifs',
        width=700,
        height=600
    )
    
    return fig
