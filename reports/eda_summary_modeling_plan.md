
# Résumé EDA & Stratégie de Modélisation (CAMELYON17-WILDS)

Ce document synthétise les découvertes de la « Deep EDA » et définit la stratégie pour la phase de modélisation.

## 1. Synthèse des Données

| Métrique | Valeur | Observation |
|---|---|---|
| **Total Patchs** | 455,954 | Données massives, cohérent avec métadonnées. |
| **Dimensions** | 96 x 96 px | Petite résolution. Nécessite adaptation modèle. |
| **Qualité** | Excellente | < 0.2% de patchs vides ou flous. |
| **Patients** | 43 | Nombre faible par rapport au volume de patchs (~10k/patient). |
| **Hôpitaux** | 5 (Centers 0-4) | Source majeure de variation. |
| **Déséquilibre** | 50/50 (H:T) | **Équilibré** dans ce dataset (vérifié). Pas de correction nécessaire. |

## 2. Analyse du Domain Shift (Hôpitaux)

Des différences significatives de colorimétrie ont été observées entre les hôpitaux :

- **Hôpital 1** : Très sombre (Moyenne Luma: 132).
- **Hôpital 4** : Très clair (Moyenne Luma: 186).
- **Shift** : L'écart-type des moyennes RGB entre hôpitaux est élevé.

**Implication** : Le modèle risque d'apprendre "Hôpital X = Tumeur" si la prévalence varie (ce qui n'est pas le cas ici, 50/50 partout), ou simplement d'échouer sur l'hôpital de test (OOD) à cause du changement de distribution des inputs.

## 3. Stratégie de Modélisation

### 3.1 Architecture du Modèle

**Recommendation : CNN "From Scratch" ou ResNet18 Adapté**

Étant donné la résolution 96x96 :

1. **Option A (Retenue)** : **Custom SimpleCNN**.
    - Architecture légère (4-5 blocs Conv).
    - Rapide à entraîner.
    - Pas d'artefacts d'upsampling.
    - Suffisant pour classifier des textures/cellules.
2. **Option B** : **ResNet18 (modifié)**.
    - Supprimer le premier MaxPool pour garder la résolution spatiale.
    - Rempalcer la couche FC finale.
    - Potentiellement plus puissant mais plus lourd.

### 3.2 Prétraitement & Augmentation

Pour contrer le Domain Shift :

- **Normalisation** : `z-score` pixel-wise (classique).
- **Augmentation Couleur (CRITIQUE)** :
  - `ColorJitter` (Brightness, Contrast, Saturation, Hue).
  - `RandomGrayscale` (faible probabilité).
  - *Idéalement* : Stain Normalization (Macenko), mais coûteux "on-the-fly". On commencera par ColorJitter fort.
- **Augmentation Géométrique** :
  - `RandomHorizontalFlip`, `RandomVerticalFlip`.
  - `RandomRotation` (90°).

### 3.3 Gestion des Splits

- Utiliser les splits fournis par WILDS (`metadata['split']`).
- **Train** : Split 0 (90%) - Probablement Train + ID-Val.
- **Test** : Split 1 (10%) - Probablement OOD-Test (Hôpital inconnu ?).
- *Action* : Lors du training, on vérifiera si le dataset de validation contient des hôpitaux vus ou non vus.

### 3.4 Fonction de Coût

- **BCEWithLogitsLoss**.
- Pas de pondération (`pos_weight`) nécessaire car équilibre 50/50 confirmé.

## 4. Next Steps

1. Implémenter le `SimpleCNN` (input 96x96).
2. Créer script d'entraînement `train_model.py`.
3. Lancer entraînement baseline avec Augmentation standard.
4. Évaluer sur Test OOD.
