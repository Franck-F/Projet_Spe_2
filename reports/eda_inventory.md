# Rapport d'Inventaire EDA - CAMELYON17

## 1. Structure des Fichiers
- **Root**: `data\raw\wilds\camelyon17_v1.0`
- **Dossier Patches**: ✅ Présent (Structure hiérarchique détectée)
- **Nombre total de patchs (.png)**: 455954
- **Fichiers CSV trouvés**: metadata.csv

## 2. Analyse Metadata
- **Dimensions**: (455954, 8)
- **Colonnes**: patient, node, x_coord, y_coord, tumor, slide, center, split
- **Valeurs manquantes**: ✅ Aucune
- **Cohérence**: ✅ 455954 entrées métadonnées == 455954 fichiers physiques

## 3. Analyse des Splits
- **Split 0**: 410359 patchs (90.0%)
- **Split 1**: 45595 patchs (10.0%)