"""
Module de chargement des données CAMELYON17
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class WSILoader:
    """
    Chargeur pour les Whole Slide Images
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Chemin vers les données brutes
        """
        self.data_path = Path(data_path)
        
    def load_wsi(self, wsi_id: str):
        """
        Charge une WSI complète
        
        Args:
            wsi_id: Identifiant de la WSI
            
        Returns:
            WSI chargée
        """
        # TODO: Implémenter avec OpenSlide
        raise NotImplementedError("À implémenter")
        
    def extract_patches(self, wsi, patch_size: Tuple[int, int] = (224, 224)):
        """
        Extrait des patchs d'une WSI
        
        Args:
            wsi: WSI source
            patch_size: Taille des patchs
            
        Returns:
            Liste de patchs
        """
        # TODO: Implémenter extraction de patchs
        raise NotImplementedError("À implémenter")


class PatchLoader:
    """
    Chargeur pour les patchs pré-extraits
    """
    
    def __init__(self, patch_dir: str):
        """
        Args:
            patch_dir: Dossier contenant les patchs
        """
        self.patch_dir = Path(patch_dir)
        
    def load_patch(self, patch_id: str) -> np.ndarray:
        """
        Charge un patch individuel
        
        Args:
            patch_id: Identifiant du patch
            
        Returns:
            Patch sous forme de numpy array
        """
        # TODO: Implémenter chargement de patch
        raise NotImplementedError("À implémenter")
        
    def get_patch_label(self, patch_id: str) -> int:
        """
        Récupère le label d'un patch
        
        Args:
            patch_id: Identifiant du patch
            
        Returns:
            Label (0: normal, 1: tumoral)
        """
        # TODO: Implémenter récupération de label
        raise NotImplementedError("À implémenter")
