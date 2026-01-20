# Stratégie d'Obtention du Dataset CAMELYON17

## Contexte

**Objectif** : Obtenir un dataset représentatif pour le projet CAMELYON17 avec labels pN  
**Contrainte** : Éviter le téléchargement de 2+ TB de WSI brutes

---

## Exploration Initiale

### Tentative 1 : Téléchargement Direct depuis AWS S3

**Source** : `s3://camelyon-dataset/CAMELYON17/`

**Découvertes** :

- 1000 WSI de 5 hôpitaux néerlandais
- Labels pN disponibles dans `example.csv`
- Stades : pN0, pN0(i+), pN1mi, pN1, pN2
- Taille prohibitive : **~2.25 TB** pour 150 patients

**Conclusion** : Approche non viable pour un projet académique

---

## Solution Retenue : WILDS Dataset

### Pourquoi WILDS ?

**WILDS** (Wild Datasets) est une bibliothèque Stanford qui fournit des datasets standardisés pour l'apprentissage robuste.

**Avantages** :

1. **Léger** : ~10 GB au lieu de 2+ TB
2. **Pré-traité** : Patchs 96×96 déjà extraits
3. **Complet** : Labels pN + métadonnées hôpital
4. **Standardisé** : Splits train/val/test définis
5. **Reproductible** : Utilisé dans la recherche académique

### Installation

```bash
# Ajouter WILDS au projet
uv add wilds

# Télécharger le dataset
uv run python scripts/download_wilds_camelyon17.py
```

### Structure du Dataset WILDS

```
data/raw/wilds/camelyon17_v1.0/
├── patches/               # Patchs 96×96 RGB
├── metadata.csv           # Labels + métadonnées
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

---

## Caractéristiques du Dataset

### Taille et Composition

- **Taille totale** : ~10 GB
- **Nombre de patchs** : ~450,000
- **Taille des patchs** : 96 × 96 × 3 (RGB)
- **Format** : PNG

### Labels Disponibles

**Niveau Patch** :

- `0` : Normal
- `1` : Tumoral

**Niveau Patient (pN stages)** :

- `pN0` : Aucun ganglion atteint
- `pN0(i+)` : Cellules tumorales isolées
- `pN1mi` : Micrométastases
- `pN1` : 1-3 ganglions positifs
- `pN2` : 4-9 ganglions positifs

### Métadonnées

- **Hospital** : ID de l'hôpital (0-4) pour analyse de fairness
- **Patient** : ID du patient
- **Node** : ID du ganglion lymphatique
- **Slide** : Nom de la lame

---

## Validation de la Stratégie

### Critères de Réussite

- [x] Dataset obtenu (< 50 GB)
- [x] Labels pN disponibles
- [x] Métadonnées hôpital pour fairness
- [x] Splits train/val/test définis
- [x] Compatible avec notre architecture

### Limitations Acceptées

- Patchs 96×96 (au lieu de 224×224)
  - **Solution** : Redimensionner ou fine-tuner sur 96×96
- Pas de pN3 dans le dataset
  - **Solution** : Travailler avec 5 classes (pN0-pN2)

---

## Références

- **WILDS** : <https://wilds.stanford.edu/datasets/#camelyon17>
- **Paper** : Koh et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts", ICML 2021
- **CAMELYON17** : <https://camelyon17.grand-challenge.org/>
