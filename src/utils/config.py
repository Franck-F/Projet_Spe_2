"""
Gestion de la configuration du projet
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Charge le fichier de configuration YAML
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Fichier de configuration non trouvé: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Créer les dossiers nécessaires
    _create_directories(config)
    
    return config


def _create_directories(config: Dict[str, Any]) -> None:
    """
    Crée les dossiers nécessaires au projet
    
    Args:
        config: Dictionnaire de configuration
    """
    paths = config.get('paths', {})
    
    for path_key, path_value in paths.items():
        Path(path_value).mkdir(parents=True, exist_ok=True)
        
        # Créer les sous-dossiers pour certains chemins
        if path_key == 'data_root':
            for subdir in ['raw', 'processed', 'annotations']:
                Path(path_value, subdir).mkdir(exist_ok=True)
                
        elif path_key == 'models':
            for subdir in ['checkpoints', 'final']:
                Path(path_value, subdir).mkdir(exist_ok=True)
                
        elif path_key == 'results':
            for subdir in ['metrics', 'figures', 'predictions']:
                Path(path_value, subdir).mkdir(exist_ok=True)


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Sauvegarde la configuration dans un fichier YAML
    
    Args:
        config: Dictionnaire de configuration
        output_path: Chemin de sortie
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_device(config: Dict[str, Any]) -> str:
    """
    Détermine le device à utiliser (cuda/cpu)
    
    Args:
        config: Dictionnaire de configuration
        
    Returns:
        Device string ('cuda' ou 'cpu')
    """
    import torch
    
    device = config.get('resources', {}).get('device', 'cuda')
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA non disponible, utilisation du CPU")
        return 'cpu'
    
    return device
