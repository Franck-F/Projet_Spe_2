"""
Télécharge les métadonnées CAMELYON17 depuis AWS S3
"""
import subprocess
import pandas as pd
from pathlib import Path

# Créer les dossiers
Path('data/raw/metadata').mkdir(parents=True, exist_ok=True)

# Télécharger les métadonnées
metadata_files = [
    'patient_labels.csv',
    'slide_info.csv',
    'hospital_mapping.csv'
]

for file in metadata_files:
    s3_path = f's3://camelyon-dataset/CAMELYON17/metadata/{file}'
    local_path = f'data/raw/metadata/{file}'
    
    cmd = [
        'aws', 's3', 'cp',
        '--no-sign-request',
        s3_path,
        local_path
    ]
    
    print(f"Téléchargement de {file}...")
    subprocess.run(cmd, check=True)
    print(f" {file} téléchargé")

# Charger et afficher les statistiques
print("\n=== Statistiques du Dataset ===")
patient_labels = pd.read_csv('data/raw/metadata/patient_labels.csv')

print(f"Nombre total de patients : {len(patient_labels)}")
print(f"\nDistribution des stades pN :")
print(patient_labels['pn_stage'].value_counts().sort_index())

print(f"\nDistribution par hôpital :")
print(patient_labels['hospital'].value_counts().sort_index())