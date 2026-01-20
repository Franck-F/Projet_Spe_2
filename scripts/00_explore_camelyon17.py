"""
Script pour explorer la structure de CAMELYON17 sur S3
"""
import subprocess
from pathlib import Path

print("=== Exploration Détaillée de CAMELYON17 ===\n")

# Créer le dossier
Path('data/raw/metadata').mkdir(parents=True, exist_ok=True)

# 1. Lister le contenu de CAMELYON17
print("1. Contenu du dossier CAMELYON17...")
cmd = ['aws', 's3', 'ls', '--no-sign-request', 's3://camelyon-dataset/CAMELYON17/']

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)

# 2. Rechercher des fichiers CSV/métadonnées
print("\n2. Recherche de fichiers CSV et métadonnées...")
cmd = ['aws', 's3', 'ls', '--no-sign-request', '--recursive', 
       's3://camelyon-dataset/CAMELYON17/']

result = subprocess.run(cmd, capture_output=True, text=True)

# Filtrer les fichiers intéressants
lines = result.stdout.strip().split('\n')
metadata_files = []

for line in lines:
    if any(ext in line.lower() for ext in ['.csv', '.xlsx', '.json', '.txt', 'readme', 'label', 'stage', 'patient']):
        metadata_files.append(line)
        print(line)

# Sauvegarder la liste complète
output_file = 'data/raw/metadata/camelyon17_files.txt'
with open(output_file, 'w') as f:
    f.write(result.stdout)

print(f"\n✅ Liste complète sauvegardée : {output_file}")
print(f"✅ {len(lines)} fichiers trouvés dans CAMELYON17")
print(f"✅ {len(metadata_files)} fichiers de métadonnées potentiels")

# 3. Chercher spécifiquement les labels pN
print("\n3. Recherche spécifique de labels pN...")
pn_keywords = ['pn', 'stage', 'label', 'patient', 'clinical']

for keyword in pn_keywords:
    matching = [line for line in lines if keyword in line.lower()]
    if matching:
        print(f"\n--- Fichiers contenant '{keyword}' ---")
        for match in matching[:10]:  # Limiter à 10
            print(match)
