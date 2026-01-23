# Rapport EDA : Qualité des Patchs

> Analyse basée sur un échantillon de 5000 patchs.

## 1. Dimensions et Format

- **(96, 96, 3)**: 5000 patchs (100.0%)
- Dimensions constantes.

## 2. Détection de Fond Blanc (Background)

- **Candidats 'Fond Blanc/Vide'** (Mean > 210, Std < 15):
  - Nombre: 7
  - Pourcentage: 0.14%

## 3. Analyse du Flou (Blur)

- **Candidats 'Flous'** (Laplacian Var < 100):
  - Nombre: 16
  - Pourcentage: 0.32%
- **Statistiques Blur Variance**:
  - Moyenne: 2615.2
  - Médiane: 2304.0

![Distributions](reports/figures/patch_quality_dist.png)
