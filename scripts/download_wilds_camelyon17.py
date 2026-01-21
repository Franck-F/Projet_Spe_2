"""
Télécharge le dataset CAMELYON17 via WILDS
Taille : ~50 GB
Temps estimé : 30-60 minutes selon la connexion
"""
from wilds import get_dataset
from pathlib import Path
import time


print("Téléchargement du Dataset CAMELYON17 via WILDS")

# Créer le dossier de destination
data_dir = Path('data/raw/wilds')
data_dir.mkdir(parents=True, exist_ok=True)

print(f"\nDossier de destination : {data_dir.absolute()}")

# Démarrer le téléchargement
start_time = time.time()

try:
    print("Téléchargement en cours...")
    
    dataset = get_dataset(
        dataset="camelyon17",
        download=True,
        root_dir=str(data_dir)
    )
    
    elapsed_time = time.time() - start_time
      
    print("Téléchargement terminé !")
        
    print(f"\nTemps écoulé : {elapsed_time/60:.1f} minutes")
    print(f"Nombre total d'images : {len(dataset):,}")
    print(f"Taille des patchs : {dataset[0][0].shape}")
    print(f"Nombre de classes : {dataset.n_classes}")
    
    # Informations sur les splits
    print(f"\n--- Splits disponibles ---")
    for split_name in ['train', 'val', 'test', 'id_val', 'id_test']:
        if split_name in dataset.split_dict:
            split_size = len(dataset.split_dict[split_name])
            print(f"{split_name:10s} : {split_size:,} patchs")
    
    # Informations sur les métadonnées
    print(f"\n--- Métadonnées ---")
    metadata_df = dataset.metadata_df
    print(f"Colonnes : {metadata_df.columns.tolist()}")
    
    print(f"\nDistribution par hôpital :")
    print(metadata_df['hospital'].value_counts().sort_index())
    
    print(f"\nDistribution des labels (patch-level) :")
    print(metadata_df['tumor'].value_counts())
    
    print("\nDataset prêt à l'emploi !")
    print(f"Emplacement : {data_dir.absolute()}")
    
except Exception as e:
    print(f"\nErreur lors du téléchargement : {e}")
    print("\nVérifiez :")
    print("  - Votre connexion Internet")
    print("  - L'espace disque disponible (>50 GB)")
    print("  - Les permissions d'écriture")
    raise
