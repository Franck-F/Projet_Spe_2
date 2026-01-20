"""
Visualisations des résultats de modèles
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def plot_model_comparison(results_df: pd.DataFrame,
                          metric: str = 'accuracy') -> go.Figure:
    """
    Compare les performances de différents modèles
    
    Args:
        results_df: DataFrame avec colonnes ['model', 'metric', 'value']
        metric: Métrique à visualiser
        
    Returns:
        Figure Plotly
    """
    df_metric = results_df[results_df['metric'] == metric]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df_metric['model'],
            y=df_metric['value'],
            text=df_metric['value'].round(3),
            textposition='auto',
            marker_color='#3498db'
        )
    ])
    
    fig.update_layout(
        title=f'Comparaison des Modèles - {metric.capitalize()}',
        xaxis_title='Modèle',
        yaxis_title=metric.capitalize(),
        width=900,
        height=500
    )
    
    return fig


def plot_metrics_radar(metrics_dict: dict, model_name: str = 'Model') -> go.Figure:
    """
    Crée un graphique radar des métriques
    
    Args:
        metrics_dict: Dictionnaire de métriques
        model_name: Nom du modèle
        
    Returns:
        Figure Plotly
    """
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=model_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f'Métriques - {model_name}',
        width=600,
        height=600
    )
    
    return fig


def plot_prediction_distribution(y_proba: np.ndarray,
                                 y_true: np.ndarray) -> go.Figure:
    """
    Visualise la distribution des probabilités prédites
    
    Args:
        y_proba: Probabilités prédites
        y_true: Labels réels
        
    Returns:
        Figure Plotly
    """
    df = pd.DataFrame({
        'probability': y_proba[:, 1],  # Probabilité classe positive
        'true_label': y_true
    })
    
    fig = go.Figure()
    
    for label in [0, 1]:
        data = df[df['true_label'] == label]['probability']
        
        fig.add_trace(go.Histogram(
            x=data,
            name=f'Classe {label}',
            opacity=0.7,
            nbinsx=50
        ))
    
    fig.update_layout(
        title='Distribution des Probabilités Prédites',
        xaxis_title='Probabilité',
        yaxis_title='Fréquence',
        barmode='overlay',
        width=800,
        height=500
    )
    
    return fig


def plot_calibration_curve(y_true: np.ndarray,
                           y_proba: np.ndarray,
                           n_bins: int = 10) -> go.Figure:
    """
    Trace la courbe de calibration
    
    Args:
        y_true: Labels réels
        y_proba: Probabilités prédites
        n_bins: Nombre de bins
        
    Returns:
        Figure Plotly
    """
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba[:, 1], n_bins=n_bins, strategy='uniform'
    )
    
    fig = go.Figure()
    
    # Courbe de calibration
    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode='lines+markers',
        name='Modèle',
        line=dict(color='#3498db', width=2)
    ))
    
    # Ligne parfaite
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Parfaitement Calibré',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title='Courbe de Calibration',
        xaxis_title='Probabilité Prédite Moyenne',
        yaxis_title='Fraction de Positifs',
        width=700,
        height=600
    )
    
    return fig


def plot_feature_importance(feature_names: list,
                            importances: np.ndarray,
                            top_n: int = 20) -> go.Figure:
    """
    Visualise l'importance des features
    
    Args:
        feature_names: Noms des features
        importances: Importances
        top_n: Nombre de top features à afficher
        
    Returns:
        Figure Plotly
    """
    # Trier par importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker_color='#3498db'
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Features les Plus Importantes',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        width=800
    )
    
    return fig
