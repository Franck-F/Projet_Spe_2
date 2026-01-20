"""
Télécharge les fichiers de métadonnées clés de CAMELYON17
"""
import subprocess
from pathlib import Path

print("=== Téléchargement des Métadonnées CAMELYON17 ===\n")

# Créer le dossier
Path('data/raw/metadata/camelyon17').mkdir(parents=True, exist_ok=True)

# Fichiers à télécharger
files_to_download = [
    'CAMELYON17/README.md',
    'CAMELYON17/evaluation/example.csv',
    'CAMELYON17/evaluation/evaluate.py',
]

for file_path in files_to_download:
    s3_path = f's3://camelyon-dataset/{file_path}'
    local_path = f'data/raw/metadata/camelyon17/{Path(file_path).name}'
    
    cmd = ['aws', 's3', 'cp', '--no-sign-request', s3_path, local_path]
    
    print(f"Téléchargement de {Path(file_path).name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {Path(file_path).name} téléchargé")
    else:
        print(f"❌ Erreur: {result.stderr}")

print("\n=== Téléchargement terminé ===")
