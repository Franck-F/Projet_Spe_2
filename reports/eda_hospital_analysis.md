# Rapport EDA : Analyse par Hôpital & Domain Shift

## 1. Distribution Quantitative par Hôpital
| Hôpital (Center) | Patchs | % Patchs | Patients | Patchs/Patient (Moy) |
|---|---|---|---|---|
| 0 | 59,436 | 13.0% | 7 | 8490.9 |
| 1 | 34,904 | 7.7% | 8 | 4363.0 |
| 2 | 85,054 | 18.7% | 9 | 9450.4 |
| 3 | 129,838 | 28.5% | 10 | 12983.8 |
| 4 | 146,722 | 32.2% | 9 | 16302.4 |

## 2. Déséquilibre des Labels (Normal vs Tumeur)
| Hôpital | Normal (0) | Tumeur (1) | % Tumeur | Ratio N:T |
|---|---|---|---|---|
| 0 | 29,718 | 29,718 | 50.00% | 1.0:1 |
| 1 | 17,452 | 17,452 | 50.00% | 1.0:1 |
| 2 | 42,527 | 42,527 | 50.00% | 1.0:1 |
| 3 | 64,919 | 64,919 | 50.00% | 1.0:1 |
| 4 | 73,361 | 73,361 | 50.00% | 1.0:1 |

## 3. Analyse du Domain Shift (Statistiques RGB)
> Basé sur un échantillon aléatoire de 500 patchs par hôpital.

| Hôpital | Mean R | Mean G | Mean B | Luma (Brightness) |
|---|---|---|---|---|
| 0 | 187.3 | 154.6 | 178.9 | 167.1 |
| 1 | 155.5 | 117.6 | 151.7 | 132.8 |
| 2 | 171.7 | 122.2 | 187.8 | 144.5 |
| 3 | 172.8 | 123.9 | 157.3 | 142.3 |
| 4 | 204.5 | 172.5 | 209.5 | 186.3 |

## 4. Observations sur le Shift

- **Hôpital le plus divergent** : Center 4 (Distance au centre moyen: 53.96)
- Les différences de luminosité et de balance des couleurs indiquent la nécessité d'une **Normalisation** ou d'une **Augmentation de couleur** robuste.