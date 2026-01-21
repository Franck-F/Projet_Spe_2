"""
Projet Spe 2 - Détection de Métastases CAMELYON17

Package principal pour la détection automatique de métastases ganglionnaires
dans le cancer du sein à partir d'images histopathologiques.
"""

__version__ = "0.1.0"
__author__ = "Votre Équipe"

from src.utils.config import load_config
from src.utils.logger import setup_logger

__all__ = ["load_config", "setup_logger"]
