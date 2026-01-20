from torch.utils.data import DataLoader
from src.data.wilds_dataset import WildsCAMELYON17Dataset
from torchvision import transforms
import torch

def get_wilds_dataloaders(
    root_dir='data/raw/wilds',
    batch_size=32,
    num_workers=4,
    pin_memory=True
):
    """
    Crée les dataloaders pour train/val/test du dataset WILDS CAMELYON17.
    
    Args:
        root_dir (str): Chemin racine du dataset WILDS
        batch_size (int): Taille du batch
        num_workers (int): Nombre de workers pour le chargement des données
        pin_memory (bool): Utiliser la mémoire épinglée (recommandé pour GPU)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print(f"Création des DataLoaders WILDS (batch_size={batch_size})...")

    # Transformations
    # Normalisation ImageNet standard (recommandée pour transfer learning)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    # Augmentation pour le train
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        normalize
    ])
    
    # Juste normalisation pour val/test
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Création des datasets
    try:
        print("Initialisation du dataset TRAIN...")
        train_dataset = WildsCAMELYON17Dataset(
            root_dir=root_dir, 
            split='train', 
            transform=train_transform,
            download=False
        )
        
        print("Initialisation du dataset VAL (id_val)...")
        # Note: WILDS a 'id_val' (in-distribution validation) et 'val' (OOD validation)
        # Pour le développement standard, on utilise souvent 'id_val' pour monitorer l'entraînement
        # Mais pour WILDS benchmark, on regarde aussi 'val' (OOD).
        # Ici on charge 'id_val' comme validation par défaut.
        val_dataset = WildsCAMELYON17Dataset(
            root_dir=root_dir, 
            split='id_val', 
            transform=eval_transform,
            download=False
        )
        
        print("Initialisation du dataset TEST (test)...")
        # 'test' dans WILDS CAMELYON17 est le test set officiel (hôpitaux inédits)
        test_dataset = WildsCAMELYON17Dataset(
            root_dir=root_dir, 
            split='test', 
            transform=eval_transform,
            download=False
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation des datasets : {e}")
        raise

    # Création des DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"✅ DataLoaders créés :")
    print(f"  - Train : {len(train_loader)} batchs")
    print(f"  - Val   : {len(val_loader)} batchs")
    print(f"  - Test  : {len(test_loader)} batchs")
    
    return train_loader, val_loader, test_loader

# Bloc de test simple
if __name__ == "__main__":
    try:
        # Test rapide (nécessite que le dataset soit extrait)
        train_loader, _, _ = get_wilds_dataloaders(batch_size=4, num_workers=0)
        batch = next(iter(train_loader))
        print("\nTest chargement batch réussi :")
        print(f"  - Images shape : {batch['image'].shape}")
        print(f"  - Labels shape : {batch['label'].shape}")
        print(f"  - Metadata keys : {batch.keys()}")
    except Exception as e:
        print(f"\n⚠️ Test ignoré (dataset non prêt ou erreur) : {e}")
