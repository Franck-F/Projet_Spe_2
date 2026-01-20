"""
Configuration du système de logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "camelyon17",
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Configure et retourne un logger
    
    Args:
        name: Nom du logger
        log_file: Chemin du fichier de log (optionnel)
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger configuré
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Éviter les doublons de handlers
    if logger.handlers:
        return logger
    
    # Format des messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier (optionnel)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TqdmLoggingHandler(logging.Handler):
    """
    Handler pour intégrer logging avec tqdm
    Évite les conflits d'affichage entre logs et barres de progression
    """
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
