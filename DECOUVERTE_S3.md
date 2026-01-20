# Découverte de la Structure S3 CAMELYON

## ✅ Ce que nous avons trouvé

### Bucket S3 : `s3://camelyon-dataset/`

**Structure découverte** :

```
s3://camelyon-dataset/
├── CAMELYON16/
│   ├── README.md
│   ├── annotations/          # Fichiers XML d'annotations
│   ├── background_tissue/    # Masques de tissus
│   ├── images/               # WSI (.tif) - TRÈS VOLUMINEUX
│   ├── evaluation/
│   └── checksums.md5
│
└── CAMELYON17/
    ├── (structure à explorer)
    └── ...
```

### Tailles des Fichiers

**CAMELYON16 WSI** (exemples) :

- `normal_001.tif` : 1.2 GB
- `normal_003.tif` : 2.1 GB  
- `test_042.tif` : 3.4 GB

**⚠️ Problème** : Les WSI complètes sont ÉNORMES (1-3 GB par fichier)

---

## 🚨 Changement de Stratégie

### Option 1 : Utiliser CAMELYON16 (Plus Simple)

**Avantages** :

- Structure claire et documentée
- Annotations XML disponibles
- README.md avec instructions

**Inconvénients** :

- Pas de labels pN (seulement normal/tumor)
- Ne correspond pas exactement au sujet (CAMELYON17)

### Option 2 : Explorer CAMELYON17 en Détail

**À faire** :

1. Lister le contenu de `CAMELYON17/`
2. Chercher les métadonnées (labels pN)
3. Identifier la structure des données

### Option 3 : Utiliser des Patchs Pré-extraits

**Rechercher** :

- Datasets de patchs déjà extraits
- PCam (PatchCamelyon) - version simplifiée
- Autres sources (Kaggle, Papers with Code)

---

## 📋 Plan d'Action Recommandé

### Étape 1 : Explorer CAMELYON17 en Détail

```bash
# Lister le contenu de CAMELYON17
aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON17/ --recursive | head -100
```

### Étape 2 : Chercher les Métadonnées

Fichiers à chercher :

- `patient_labels.csv` ou `.xlsx`
- `stage_labels.csv`
- `pn_stages.csv`
- `README.md` ou `README.txt`

### Étape 3 : Alternative - Utiliser PCam

**PCam (PatchCamelyon)** :

- Dataset de patchs 96×96 déjà extraits
- ~300,000 patchs
- Labels binaires (normal/tumor)
- Taille : ~7 GB (gérable)

**Source** : <https://github.com/basveeling/pcam>

---

## 💡 Recommandation Immédiate

**Je recommande d'explorer CAMELYON17 d'abord** pour voir si :

1. Les labels pN sont disponibles
2. Des patchs pré-extraits existent
3. La structure est utilisable

**Si CAMELYON17 n'a pas les labels pN** :

- Utiliser CAMELYON16 pour la partie technique
- Simuler les stades pN basés sur le % de patchs tumoraux
- Documenter cette limitation dans le rapport

---

## 🔄 Prochaines Étapes

1. **Explorer CAMELYON17** en détail
2. **Chercher les métadonnées** avec labels pN
3. **Décider** : CAMELYON17 complet, CAMELYON16, ou PCam
4. **Adapter** les scripts de téléchargement

**Voulez-vous que je continue l'exploration de CAMELYON17 ?**
