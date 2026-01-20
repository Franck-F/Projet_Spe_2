# Guide de Téléchargement - Dataset CAMELYON17 depuis AWS S3

## 📦 Informations AWS

**Bucket S3** : `s3://camelyon-dataset`  
**Région** : `us-west-2`  
**Accès** : Public (pas de compte AWS requis)

## 🎯 Objectif

Télécharger un **échantillon représentatif** de 120-150 patients selon la stratégie de sous-échantillonnage définie.

---

## 📋 Étape 1 : Installation AWS CLI

### Windows

```powershell
# Télécharger AWS CLI v2
# https://awscli.amazonaws.com/AWSCLIV2.msi

# Ou via winget
winget install Amazon.AWSCLI

# Vérifier l'installation
aws --version
```

### Linux/macOS

```bash
# Via curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Vérifier
aws --version
```

---

## Étape 2 : Explorer le Dataset

### 2.1 Lister le Contenu du Bucket

```bash
# Lister les dossiers principaux
aws s3 ls --no-sign-request s3://camelyon-dataset/

# Exemple de sortie attendue :
# PRE CAMELYON16/
# PRE CAMELYON17/
# PRE annotations/
# PRE metadata/
```

### 2.2 Explorer CAMELYON17

```bash
# Lister le contenu CAMELYON17
aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON17/ --recursive

# Sauvegarder la liste dans un fichier
aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON17/ --recursive > data/raw/camelyon17_file_list.txt
```

---

## Étape 3 : Télécharger les Métadonnées

**Script** : `scripts/01_download_metadata.py`

```python
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
```

**Exécution** :

```bash
uv run python scripts/01_download_metadata.py
```

---

## Étape 4 : Sélection Stratifiée des Patients

**Script** : `scripts/02_select_patients.py`

```python
"""
Sélection stratifiée de 150 patients
Critères : hospital × pn_stage
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Paramètres
N_PATIENTS_TARGET = 150
RANDOM_SEED = 42

# Charger les métadonnées
df = pd.read_csv('data/raw/metadata/patient_labels.csv')

print(f"Dataset complet : {len(df)} patients")
print(f"Objectif : {N_PATIENTS_TARGET} patients")

# Stratification : hospital × pn_stage
selected_patients = []

# Calculer le nombre de patients par groupe
n_hospitals = df['hospital'].nunique()
n_stages = df['pn_stage'].nunique()
n_per_group = N_PATIENTS_TARGET // (n_hospitals * n_stages)

print(f"\nNombre de patients par groupe (hospital × pN) : ~{n_per_group}")

# Sélection stratifiée
np.random.seed(RANDOM_SEED)

for hospital in sorted(df['hospital'].unique()):
    for pn_stage in sorted(df['pn_stage'].unique()):
        # Filtrer le sous-groupe
        subset = df[(df['hospital'] == hospital) & (df['pn_stage'] == pn_stage)]
        
        if len(subset) == 0:
            continue
        
        # Nombre à sélectionner
        n_select = min(len(subset), n_per_group)
        
        # Échantillonnage aléatoire
        sample = subset.sample(n=n_select, random_state=RANDOM_SEED)
        selected_patients.append(sample)
        
        print(f"Hôpital {hospital}, pN{pn_stage}: {n_select}/{len(subset)} patients sélectionnés")

# Combiner
final_selection = pd.concat(selected_patients, ignore_index=True)

print(f"\n=== Sélection Finale ===")
print(f"Total : {len(final_selection)} patients")

print(f"\nDistribution par hôpital :")
print(final_selection['hospital'].value_counts().sort_index())

print(f"\nDistribution par stade pN :")
print(final_selection['pn_stage'].value_counts().sort_index())

print(f"\nDistribution croisée (hospital × pN) :")
print(pd.crosstab(final_selection['hospital'], final_selection['pn_stage']))

# Sauvegarder
output_path = 'data/processed/selected_patients.csv'
Path('data/processed').mkdir(parents=True, exist_ok=True)
final_selection.to_csv(output_path, index=False)

print(f"\n Sélection sauvegardée : {output_path}")
```

**Exécution** :

```bash
uv run python scripts/02_select_patients.py
```

---

## 📋 Étape 5 : Téléchargement des WSI Sélectionnées

**Script** : `scripts/03_download_selected_wsi.py`

```python
"""
Télécharge uniquement les WSI des patients sélectionnés
"""
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Charger la sélection
selected = pd.read_csv('data/processed/selected_patients.csv')

print(f"Téléchargement de {len(selected)} patients...")

# Créer le dossier de destination
wsi_dir = Path('data/raw/wsi')
wsi_dir.mkdir(parents=True, exist_ok=True)

# Statistiques
total_size_gb = 0
failed_downloads = []

# Pour chaque patient
for idx, row in tqdm(selected.iterrows(), total=len(selected)):
    patient_id = row['patient_id']
    hospital = row['hospital']
    
    # Construire le chemin S3
    # Format typique : CAMELYON17/center_X/patient_XXX.tif
    s3_path = f's3://camelyon-dataset/CAMELYON17/center_{hospital}/{patient_id}.tif'
    local_path = wsi_dir / f'{patient_id}.tif'
    
    # Vérifier si déjà téléchargé
    if local_path.exists():
        print(f"⏭️  {patient_id} déjà téléchargé")
        continue
    
    # Télécharger
    cmd = [
        'aws', 's3', 'cp',
        '--no-sign-request',
        s3_path,
        str(local_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Calculer la taille
        size_mb = local_path.stat().st_size / (1024 * 1024)
        total_size_gb += size_mb / 1024
        
        print(f"✅ {patient_id} téléchargé ({size_mb:.1f} MB)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur pour {patient_id}: {e}")
        failed_downloads.append(patient_id)

# Résumé
print(f"\n=== Résumé du Téléchargement ===")
print(f"Patients téléchargés : {len(selected) - len(failed_downloads)}/{len(selected)}")
print(f"Taille totale : {total_size_gb:.2f} GB")

if failed_downloads:
    print(f"\n⚠️  Échecs ({len(failed_downloads)}) :")
    for patient_id in failed_downloads:
        print(f"  - {patient_id}")
    
    # Sauvegarder la liste des échecs
    pd.DataFrame({'patient_id': failed_downloads}).to_csv(
        'data/processed/failed_downloads.csv', index=False
    )
```

**Exécution** :

```bash
uv run python scripts/03_download_selected_wsi.py
```

**⚠️ Attention** : Ce téléchargement peut prendre plusieurs heures selon votre connexion.

---

## 📋 Étape 6 : Téléchargement Parallèle (Optionnel)

Pour accélérer le téléchargement, utilisez le téléchargement parallèle :

**Script** : `scripts/03b_download_parallel.py`

```python
"""
Téléchargement parallèle avec multiprocessing
"""
import subprocess
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

def download_patient(args):
    """Télécharge un patient"""
    patient_id, hospital, wsi_dir = args
    
    s3_path = f's3://camelyon-dataset/CAMELYON17/center_{hospital}/{patient_id}.tif'
    local_path = wsi_dir / f'{patient_id}.tif'
    
    if local_path.exists():
        return {'patient_id': patient_id, 'status': 'skipped', 'size_mb': 0}
    
    cmd = [
        'aws', 's3', 'cp',
        '--no-sign-request',
        s3_path,
        str(local_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        size_mb = local_path.stat().st_size / (1024 * 1024)
        return {'patient_id': patient_id, 'status': 'success', 'size_mb': size_mb}
    except Exception as e:
        return {'patient_id': patient_id, 'status': 'failed', 'error': str(e)}

# Charger la sélection
selected = pd.read_csv('data/processed/selected_patients.csv')
wsi_dir = Path('data/raw/wsi')
wsi_dir.mkdir(parents=True, exist_ok=True)

# Préparer les arguments
args_list = [
    (row['patient_id'], row['hospital'], wsi_dir)
    for _, row in selected.iterrows()
]

# Téléchargement parallèle (4 workers)
print(f"Téléchargement parallèle de {len(args_list)} patients...")

with Pool(processes=4) as pool:
    results = list(tqdm(
        pool.imap(download_patient, args_list),
        total=len(args_list)
    ))

# Analyser les résultats
success = [r for r in results if r['status'] == 'success']
failed = [r for r in results if r['status'] == 'failed']
skipped = [r for r in results if r['status'] == 'skipped']

total_size_gb = sum(r['size_mb'] for r in success) / 1024

print(f"\n=== Résumé ===")
print(f"✅ Succès : {len(success)}")
print(f"⏭️  Ignorés : {len(skipped)}")
print(f"❌ Échecs : {len(failed)}")
print(f"📦 Taille totale : {total_size_gb:.2f} GB")
```

**Exécution** :

```bash
uv run python scripts/03b_download_parallel.py
```

---

## 📋 Étape 7 : Vérification de l'Intégrité

**Script** : `scripts/04_verify_downloads.py`

```python
"""
Vérifie l'intégrité des WSI téléchargées
"""
import pandas as pd
from pathlib import Path
import openslide

# Charger la sélection
selected = pd.read_csv('data/processed/selected_patients.csv')
wsi_dir = Path('data/raw/wsi')

print("Vérification de l'intégrité des WSI...")

valid_wsi = []
corrupted_wsi = []

for _, row in selected.iterrows():
    patient_id = row['patient_id']
    wsi_path = wsi_dir / f'{patient_id}.tif'
    
    if not wsi_path.exists():
        print(f"❌ {patient_id}: Fichier manquant")
        continue
    
    try:
        # Essayer d'ouvrir avec OpenSlide
        slide = openslide.OpenSlide(str(wsi_path))
        
        # Vérifier les dimensions
        width, height = slide.dimensions
        
        if width > 0 and height > 0:
            valid_wsi.append({
                'patient_id': patient_id,
                'width': width,
                'height': height,
                'size_mb': wsi_path.stat().st_size / (1024 * 1024)
            })
            print(f"✅ {patient_id}: {width}×{height} pixels")
        else:
            corrupted_wsi.append(patient_id)
            print(f"⚠️  {patient_id}: Dimensions invalides")
        
        slide.close()
        
    except Exception as e:
        corrupted_wsi.append(patient_id)
        print(f"❌ {patient_id}: Erreur - {e}")

# Sauvegarder les résultats
df_valid = pd.DataFrame(valid_wsi)
df_valid.to_csv('data/processed/valid_wsi.csv', index=False)

print(f"\n=== Résumé ===")
print(f"✅ WSI valides : {len(valid_wsi)}")
print(f"❌ WSI corrompues : {len(corrupted_wsi)}")

if corrupted_wsi:
    print(f"\nWSI à retélécharger :")
    for patient_id in corrupted_wsi:
        print(f"  - {patient_id}")
```

**Exécution** :

```bash
uv run python scripts/04_verify_downloads.py
```

---

## 📊 Estimation des Ressources

### Taille du Dataset

**Par patient** :

- WSI moyenne : ~500 MB
- 150 patients : **~75 GB**

### Temps de Téléchargement

**Connexion 100 Mbps** :

- 75 GB ≈ **1h40**

**Connexion 50 Mbps** :

- 75 GB ≈ **3h20**

**Connexion 10 Mbps** :

- 75 GB ≈ **16h40**

### Espace Disque Requis

```
data/
├── raw/
│   ├── wsi/              # 75 GB (WSI)
│   └── metadata/         # < 1 MB
├── processed/
│   ├── patches/          # 20-30 GB (patchs extraits)
│   └── labels/           # < 10 MB
└── Total : ~100-110 GB
```

---

## ✅ Checklist de Téléchargement

- [ ] AWS CLI installé et vérifié
- [ ] Bucket S3 exploré
- [ ] Métadonnées téléchargées
- [ ] 150 patients sélectionnés (stratification validée)
- [ ] WSI téléchargées (75 GB)
- [ ] Intégrité vérifiée avec OpenSlide
- [ ] Documentation de la sélection créée

---

## 🔧 Dépannage

### Erreur : "Unable to locate credentials"

**Solution** : Ajouter `--no-sign-request` à toutes les commandes AWS CLI

### Erreur : "Connection timeout"

**Solution** : Réessayer ou utiliser le téléchargement parallèle

### Erreur : "File not found on S3"

**Solution** : Vérifier le chemin S3 exact avec `aws s3 ls`

### WSI corrompue

**Solution** : Retélécharger le fichier spécifique

---

## 📝 Prochaines Étapes

Après le téléchargement :

1. **Extraction de patchs** : `scripts/05_extract_patches.py`
2. **Création des labels** : `scripts/06_create_labels.py`
3. **Split train/val/test** : `scripts/07_create_splits.py`

**Voir** : `PLAN_DEVELOPPEMENT.md` - Phase 2

---

**Bon téléchargement ! 🚀**
