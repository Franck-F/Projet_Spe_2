"""
Stratégies d'agrégation patch → patient
"""

import numpy as np
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class StatisticalAggregator:
    """
    Agrégation statistique simple
    """
    
    def __init__(self, thresholds: Dict[str, float]):
        """
        Args:
            thresholds: Seuils pour classification pN
                       {'pn0': 0.0, 'pn1': 0.05, 'pn2': 0.20}
        """
        self.thresholds = thresholds
        
    def aggregate(self, patch_predictions: np.ndarray) -> int:
        """
        Agrège les prédictions au niveau patient
        
        Args:
            patch_predictions: Prédictions pour tous les patchs du patient
                              (N, 2) avec probabilités [normal, tumoral]
                              
        Returns:
            Stade pN prédit (0, 1, 2, ou 3)
        """
        # Calculer le pourcentage de patchs tumoraux
        tumor_probs = patch_predictions[:, 1]
        tumor_percentage = (tumor_probs > 0.5).mean()
        
        # Classifier selon les seuils
        if tumor_percentage <= self.thresholds['pn0']:
            return 0  # pN0
        elif tumor_percentage <= self.thresholds['pn1']:
            return 1  # pN1
        elif tumor_percentage <= self.thresholds['pn2']:
            return 2  # pN2
        else:
            return 3  # pN3


class MLAggregator:
    """
    Agrégation avec modèle ML de second niveau
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Args:
            model_type: Type de modèle ('xgboost', 'random_forest')
        """
        self.model_type = model_type
        
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
            
    def extract_features(self, patch_predictions: np.ndarray) -> np.ndarray:
        """
        Extrait des features agrégées des prédictions de patchs
        
        Args:
            patch_predictions: Prédictions pour tous les patchs
            
        Returns:
            Vecteur de features
        """
        tumor_probs = patch_predictions[:, 1]
        
        features = [
            tumor_probs.mean(),           # Probabilité moyenne
            tumor_probs.max(),            # Probabilité maximale
            tumor_probs.std(),            # Écart-type
            (tumor_probs > 0.5).mean(),   # Pourcentage de patchs tumoraux
            (tumor_probs > 0.5).sum(),    # Nombre de patchs tumoraux
            np.median(tumor_probs),       # Médiane
            np.percentile(tumor_probs, 75),  # 75e percentile
            np.percentile(tumor_probs, 90),  # 90e percentile
        ]
        
        return np.array(features)
        
    def fit(self, X: List[np.ndarray], y: np.ndarray):
        """
        Entraîne le modèle d'agrégation
        
        Args:
            X: Liste de prédictions de patchs par patient
            y: Labels patients (stades pN)
        """
        # Extraire les features pour chaque patient
        X_features = np.array([self.extract_features(x) for x in X])
        
        # Entraîner le modèle
        self.model.fit(X_features, y)
        
    def predict(self, patch_predictions: np.ndarray) -> int:
        """
        Prédit le stade pN pour un patient
        
        Args:
            patch_predictions: Prédictions pour tous les patchs
            
        Returns:
            Stade pN prédit
        """
        features = self.extract_features(patch_predictions).reshape(1, -1)
        return self.model.predict(features)[0]
