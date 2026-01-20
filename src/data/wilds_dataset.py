import torch
from torch.utils.data import Dataset
from wilds import get_dataset
from pathlib import Path
import pandas as pd
from typing import Optional, Callable, Dict, Any

class WildsCAMELYON17Dataset(Dataset):
    """
    Wrapper PyTorch pour le dataset WILDS CAMELYON17.
    
    Ce dataset contient des patchs 96x96 extraits de WSIs.
    Il fournit pour chaque patch :
    - Image (Tensor)
    - Label tumoral (0/1)
    - Métadonnées (Hôpital, Patient, Slide, Node)
    
    Args:
        root_dir (str): Répertoire racine contenant le dataset WILDS
        split (str): Split à charger ('train', 'val', 'test', 'id_val', 'id_test')
        transform (callable, optional): Transformations à appliquer aux images
        download (bool): Si True, télécharge le dataset s'il n'existe pas
    """
    def __init__(self, 
                 root_dir: str = 'data/raw/wilds', 
                 split: str = 'train', 
                 transform: Optional[Callable] = None,
                 download: bool = False):
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Charger le dataset WILDS
        self.wilds_dataset = get_dataset(
            dataset="camelyon17",
            download=download,
            root_dir=root_dir
        )
        
        # Obtenir les indices du split
        if split not in self.wilds_dataset.split_dict:
            raise ValueError(f"Split '{split}' non trouvé. Disponibles : {list(self.wilds_dataset.split_dict.keys())}")
            
        self.split_indices = self.wilds_dataset.split_dict[split]
        
        # Mapping des métadonnées (selon documentation WILDS)
        # metadata_fields = ['hospital', 'patient', 'node', 'slide', 'y']
        self.metadata_map = self.wilds_dataset.metadata_fields
        
    def __len__(self) -> int:
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retourne un item du dataset.
        
        Returns:
            dict: {
                'image': Tensor image,
                'label': Int (0=Normal, 1=Tumor),
                'hospital': Int (ID hôpital),
                'patient': Int (ID patient),
                'node': Int (ID node),
                'slide': Int (ID slide),
                'metadata': Tensor complet des métadonnées
            }
        """
        # Obtenir l'index global dans le dataset WILDS
        global_idx = self.split_indices[idx]
        
        # Charger image, label, et métadonnées
        x, y, metadata = self.wilds_dataset[global_idx]
        
        # Appliquer les transformations
        if self.transform:
            x = self.transform(x)
        
        # Construire le dictionnaire de retour
        item = {
            'image': x,
            'label': y.item(),
            'metadata': metadata
        }
        
        # Ajouter les champs de métadonnées spécifiques pour faciliter l'accès
        for i, field in enumerate(self.metadata_map):
            item[field] = metadata[i].item()
            
        return item

    def get_metadata_df(self) -> pd.DataFrame:
        """Retourne un DataFrame pandas avec les métadonnées pour ce split"""
        meta = self.wilds_dataset.metadata_df.iloc[self.split_indices].copy()
        return meta
