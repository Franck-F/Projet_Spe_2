"""
Script d'exploration amélioré du bucket S3 CAMELYON
"""
import subprocess
from pathlib import Path

# Créer les dossiers
Path('data/raw/metadata').mkdir(parents=True, exist_ok=True)

print("=== Exploration Complète du Bucket S3 CAMELYON ===\n")

# 1. Lister le contenu racine
print("1. Contenu racine du bucket...")
cmd = ['aws', 's3', 'ls', '--no-sign-request', 's3://camelyon-dataset/']

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f"Erreur: {result.stderr}")

# 2. Explorer les sous-dossiers
folders_to_check = [
    '',  # Racine
    'CAMELYON16/',
    'CAMELYON17/',
    'camelyon16/',
    'camelyon17/',
    'metadata/',
    'annotations/',
]

print("\n2. Exploration des sous-dossiers...")
for folder in folders_to_check:
    print(f"\n--- Contenu de '{folder}' ---")
    cmd = ['aws', 's3', 'ls', '--no-sign-request', f's3://camelyon-dataset/{folder}']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and result.stdout.strip():
        print(result.stdout)
    else:
        print(f"  (vide ou inexistant)")

# 3. Recherche récursive limitée
print("\n3. Liste récursive (premiers 50 fichiers)...")
cmd = ['aws', 's3', 'ls', '--no-sign-request', '--recursive', 's3://camelyon-dataset/']

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    lines = result.stdout.strip().split('\n')[:50]
    for line in lines:
        print(line)
    
    # Sauvegarder la liste complète
    output_file = 'data/raw/metadata/s3_complete_list.txt'
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    
    total_files = len(result.stdout.strip().split('\n'))
    print(f"\n✅ {total_files} fichiers au total")
    print(f"✅ Liste complète sauvegardée : {output_file}")
else:
    print(f"Erreur: {result.stderr}")

print("\n=== Exploration terminée ===")
print("\nAnalysez 'data/raw/metadata/s3_complete_list.txt' pour comprendre la structure")
