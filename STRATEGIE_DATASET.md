# Stratégie de Gestion du Dataset CAMELYON17

## 🚨 Problématiques Identifiées

### 1. Volume du Dataset

- **Problème** : Dataset CAMELYON17 complet = plusieurs centaines de GB
- **Contraintes** :
  - Espace disque limité
  - Temps de téléchargement prohibitif
  - Temps de traitement très long

### 2. Bibliothèque WILDS

- **Problème** : WILDS ne fournit pas les labels pN (pN0, pN1, pN2, pN3)
- **WILDS** : Seulement classification binaire (0: pas de cancer / 1: cancer)
- **Notre besoin** : Classification multi-classe des stades pN

### 3. Absence de Dataset Nettoyé

- **Problème** : Pas de version prétraitée sur Kaggle
- **Conséquence** : Nous devons tout faire nous-mêmes

---

## ✅ STRATÉGIE RECOMMANDÉE : Sous-échantillonnage Intelligent

### Approche Proposée

Au lieu d'utiliser le dataset complet, nous allons créer un **sous-ensemble représentatif** du dataset CAMELYON17.

### Critères de Sous-échantillonnage

#### 1. **Diversité des Centres (5 hôpitaux)**

```
Objectif : Garder la variabilité inter-hospitalière

Distribution proposée :
- Centre 1 : 20% des patients
- Centre 2 : 20% des patients
- Centre 3 : 20% des patients
- Centre 4 : 20% des patients
- Centre 5 : 20% des patients

Total : ~100-200 patients (au lieu de 1000)
```

#### 2. **Équilibre des Stades pN**

```
Objectif : Représentation équitable de chaque stade

Distribution cible :
- pN0 (pas de métastase)     : 30-40% (~40-60 patients)
- pN1 (métastase limitée)    : 25-30% (~30-40 patients)
- pN2 (métastase modérée)    : 20-25% (~25-35 patients)
- pN3 (métastase étendue)    : 15-20% (~20-30 patients)

Total : ~120-165 patients
```

#### 3. **Nombre de Patchs par Patient**

```
Objectif : Gérer le volume de données

Stratégie :
- Patients pN0 : 50-100 patchs normaux
- Patients pN1-3 : 100-200 patchs (mix normal/tumoral)

Total estimé : ~15,000-25,000 patchs (au lieu de millions)
```

---

## 📋 PLAN D'ACTION DÉTAILLÉ

### Phase 1 : Exploration et Sélection (Semaine 1)

#### 1.1 Accès au Dataset

**Options** :

**Option A : CAMELYON17 Challenge (Officiel)**

- Site : <https://camelyon17.grand-challenge.org/>
- Inscription requise
- Téléchargement sélectif possible
- **Action** : S'inscrire et explorer les métadonnées

**Option B : Kaggle (Partiel)**

- Chercher "CAMELYON17" ou "breast cancer metastasis"
- Vérifier si des sous-ensembles existent
- **Action** : Explorer Kaggle datasets

**Option C : Papers with Code**

- Chercher des implémentations existantes
- Certains auteurs partagent des sous-ensembles
- **Action** : Vérifier les repositories GitHub

#### 1.2 Télécharger les Métadonnées UNIQUEMENT

**Fichiers prioritaires** :

```
metadata/
├── patient_labels.csv      # Labels pN par patient
├── slide_info.csv          # Info sur chaque slide
├── patch_coordinates.csv   # Coordonnées des patchs
└── hospital_mapping.csv    # Mapping patient → hôpital
```

**Script à créer** : `scripts/download_metadata.py`

```python
"""
Télécharge uniquement les métadonnées CAMELYON17
"""
import pandas as pd

# Charger les métadonnées
patient_labels = pd.read_csv('metadata/patient_labels.csv')

# Afficher les statistiques
print("=== Distribution des stades pN ===")
print(patient_labels['pn_stage'].value_counts())

print("\n=== Distribution par hôpital ===")
print(patient_labels['hospital'].value_counts())
```

#### 1.3 Sélection Stratifiée des Patients

**Script** : `scripts/select_patients.py`

```python
"""
Sélection stratifiée de patients pour sous-échantillonnage
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger métadonnées
df = pd.read_csv('metadata/patient_labels.csv')

# Paramètres
N_PATIENTS_TARGET = 150  # Ajustable selon contraintes
STRATIFY_COLS = ['hospital', 'pn_stage']

# Sélection stratifiée
selected_patients = []

for hospital in df['hospital'].unique():
    for pn_stage in [0, 1, 2, 3]:
        # Filtrer
        subset = df[(df['hospital'] == hospital) & (df['pn_stage'] == pn_stage)]
        
        # Nombre à sélectionner (proportionnel)
        n_select = min(len(subset), N_PATIENTS_TARGET // 20)  # ~7-8 par groupe
        
        if len(subset) > 0:
            # Échantillonnage aléatoire
            sample = subset.sample(n=n_select, random_state=42)
            selected_patients.append(sample)

# Combiner
final_selection = pd.concat(selected_patients)

print(f"Patients sélectionnés : {len(final_selection)}")
print("\n=== Distribution finale ===")
print(final_selection.groupby(['hospital', 'pn_stage']).size())

# Sauvegarder
final_selection.to_csv('data/processed/selected_patients.csv', index=False)
```

### Phase 2 : Téléchargement Ciblé (Semaine 1-2)

#### 2.1 Télécharger UNIQUEMENT les Patients Sélectionnés

**Script** : `scripts/download_selected_wsi.py`

```python
"""
Télécharge uniquement les WSI des patients sélectionnés
"""
import pandas as pd
import requests
from tqdm import tqdm

# Charger la sélection
selected = pd.read_csv('data/processed/selected_patients.csv')

# Pour chaque patient
for idx, row in tqdm(selected.iterrows(), total=len(selected)):
    patient_id = row['patient_id']
    
    # URL du fichier (à adapter selon la source)
    url = f"https://camelyon17.org/data/{patient_id}.tif"
    
    # Télécharger
    response = requests.get(url, stream=True)
    
    # Sauvegarder
    with open(f'data/raw/{patient_id}.tif', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Téléchargé : {patient_id}")
```

**Estimation de volume** :

- 150 patients × ~500 MB/patient = **~75 GB** (gérable)
- Au lieu de 1000 patients × 500 MB = 500 GB (ingérable)

#### 2.2 Extraction de Patchs

**Script** : `scripts/extract_patches.py`

```python
"""
Extrait des patchs des WSI téléchargées
"""
import openslide
import numpy as np
from pathlib import Path

def extract_patches_from_wsi(wsi_path, n_patches=100, patch_size=224):
    """
    Extrait n_patches de taille patch_size×patch_size
    """
    # Charger WSI
    slide = openslide.OpenSlide(wsi_path)
    
    # Dimensions
    width, height = slide.dimensions
    
    patches = []
    for i in range(n_patches):
        # Coordonnées aléatoires
        x = np.random.randint(0, width - patch_size)
        y = np.random.randint(0, height - patch_size)
        
        # Extraire patch
        patch = slide.read_region((x, y), 0, (patch_size, patch_size))
        patch = np.array(patch.convert('RGB'))
        
        # Filtrer qualité (fond blanc, flou)
        if is_good_quality(patch):
            patches.append(patch)
    
    return patches

# Traiter tous les patients
for wsi_file in Path('data/raw/').glob('*.tif'):
    patches = extract_patches_from_wsi(wsi_file, n_patches=100)
    
    # Sauvegarder
    patient_id = wsi_file.stem
    save_patches(patches, f'data/processed/patches/{patient_id}/')
```

### Phase 3 : Création du Dataset Final (Semaine 2)

#### 3.1 Organisation des Données

```
data/processed/
├── patches/
│   ├── patient_001/
│   │   ├── patch_0001.png
│   │   ├── patch_0002.png
│   │   └── ...
│   ├── patient_002/
│   └── ...
├── labels/
│   ├── patch_labels.csv      # Label par patch (0: normal, 1: tumoral)
│   └── patient_labels.csv    # Label pN par patient
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

#### 3.2 Création des Labels

**Script** : `scripts/create_labels.py`

```python
"""
Crée les fichiers de labels
"""
import pandas as pd

# Labels niveau patch
patch_labels = []
for patient_dir in Path('data/processed/patches/').iterdir():
    patient_id = patient_dir.name
    pn_stage = get_pn_stage(patient_id)  # Depuis métadonnées
    
    for patch_file in patient_dir.glob('*.png'):
        # Déterminer si patch est tumoral (depuis annotations)
        is_tumor = check_if_tumor(patch_file)
        
        patch_labels.append({
            'patch_id': patch_file.stem,
            'patient_id': patient_id,
            'label': int(is_tumor),
            'pn_stage': pn_stage
        })

# Sauvegarder
pd.DataFrame(patch_labels).to_csv('data/processed/labels/patch_labels.csv', index=False)
```

---

## 📊 JUSTIFICATION DE LA STRATÉGIE

### Critères d'Évaluation Attendus

Votre stratégie de sous-échantillonnage sera évaluée sur :

#### 1. **Représentativité Statistique**

- ✅ Distribution des stades pN respectée
- ✅ Tous les centres représentés équitablement
- ✅ Variabilité inter-hospitalière préservée

#### 2. **Rigueur Méthodologique**

- ✅ Échantillonnage stratifié (pas aléatoire simple)
- ✅ Seed fixé pour reproductibilité
- ✅ Documentation complète du processus

#### 3. **Gestion du Déséquilibre**

- ✅ Stratégies pour compenser le déséquilibre
- ✅ Weighted sampling / Focal loss
- ✅ Justification des choix

#### 4. **Validation de la Généralisation**

- ✅ Test sur hôpital hold-out
- ✅ Analyse du domain shift
- ✅ Robustesse démontrée

### Documentation à Fournir

**Créer** : `reports/strategie_sous_echantillonnage.md`

**Contenu** :

```markdown
# Stratégie de Sous-échantillonnage

## 1. Contraintes
- Volume du dataset complet : 500 GB
- Ressources disponibles : 100 GB
- Temps de traitement : limité

## 2. Approche
- Sélection stratifiée de 150 patients
- Critères : hospital × pn_stage
- Extraction de 100-200 patchs/patient

## 3. Distribution Finale
[Tableaux et graphiques]

## 4. Validation
- Split train/val/test respecte la stratification
- Analyse de représentativité
- Comparaison avec dataset complet (si métadonnées disponibles)

## 5. Limites et Biais
- Réduction de la diversité
- Possibles biais de sélection
- Stratégies de mitigation
```

---

## 🎯 OBJECTIFS RÉVISÉS

### Dataset Final Cible

```
Patients : 120-150
Patchs : 15,000-25,000
Volume : 50-75 GB

Distribution :
- pN0 : 35% (~45 patients, ~5,000 patchs)
- pN1 : 30% (~40 patients, ~6,000 patchs)
- pN2 : 20% (~25 patients, ~4,000 patchs)
- pN3 : 15% (~20 patients, ~3,000 patchs)

Centres : 5 hôpitaux équilibrés
```

### Performances Attendues

**Avec dataset réduit** :

- Recall niveau patch : > 90% (au lieu de 95%)
- Accuracy niveau patient : > 75% (au lieu de 80%)
- Généralisation : À démontrer avec analyse robuste

---

## 📝 CHECKLIST DE MISE EN ŒUVRE

### Semaine 1

- [ ] S'inscrire au challenge CAMELYON17
- [ ] Télécharger les métadonnées complètes
- [ ] Analyser la distribution complète
- [ ] Implémenter `scripts/select_patients.py`
- [ ] Valider la sélection stratifiée
- [ ] Documenter la stratégie

### Semaine 2

- [ ] Télécharger les WSI sélectionnées (75 GB)
- [ ] Implémenter `scripts/extract_patches.py`
- [ ] Extraire les patchs (~20,000)
- [ ] Créer les labels
- [ ] Vérifier la qualité des patchs
- [ ] Créer les splits train/val/test

### Semaine 3

- [ ] Finaliser le dataset
- [ ] Créer le rapport de sous-échantillonnage
- [ ] Commencer l'EDA sur le dataset réduit

---

## 🔄 ALTERNATIVES SI PROBLÈMES PERSISTENT

### Plan B : Dataset Synthétique Partiel

- Utiliser WILDS pour la classification binaire
- Simuler les stades pN basés sur % de patchs tumoraux
- **Limite** : Moins réaliste médicalement

### Plan C : Collaboration

- Contacter d'autres équipes/chercheurs
- Partager un sous-ensemble déjà préparé
- **Avantage** : Gain de temps

### Plan D : Dataset Alternatif

- Chercher d'autres datasets de pathologie
- Ex : PCam (PatchCamelyon) - plus petit
- **Limite** : Pas exactement le même problème

---

## 💡 RECOMMANDATIONS FINALES

1. **Prioriser la qualité sur la quantité**
   - Mieux vaut 150 patients bien sélectionnés que 1000 mal gérés

2. **Documenter exhaustivement**
   - Chaque choix doit être justifié
   - Transparence totale sur les limites

3. **Valider la représentativité**
   - Comparer avec les statistiques du dataset complet
   - Démontrer que le sous-ensemble est représentatif

4. **Adapter les objectifs**
   - Performances légèrement inférieures acceptables
   - Focus sur la méthodologie et l'interprétabilité

5. **Communiquer tôt**
   - Informer les encadrants de la stratégie
   - Obtenir validation avant de commencer

---

**Cette stratégie transforme une contrainte (volume) en opportunité de démontrer votre rigueur méthodologique ! 🚀**
