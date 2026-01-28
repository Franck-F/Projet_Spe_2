# Documentation technique – Projet CAMELYON17

Ce document synthétise le pipeline complet développé pour détecter des métastases ganglionnaires sur les patchs histologiques du challenge CAMELYON17. Il couvre la préparation des données, la modélisation, la recherche d'hyperparamètres, l'évaluation, l'interprétabilité et la production des artefacts.

---

## 1. Périmètre et objectifs

- Objectif : classifier des patchs 224x224 (tumoral vs sain) et agréger les scores au niveau patient.
- Contraintes : hétérogénéité inter-centres (domain shift), classes déséquilibrées (~32 % positifs), budget calcul limité.
- Livrables : modèle optimisé, métriques patch/patient, traces d'entraînement, explications visuelles, scripts d'inférence.

---

## 2. Organisation du dépôt

| Chemin | Rôle |
|--------|------|
| data/raw | Métadonnées officielles WILDS, listes de fichiers, splits source. |
| data/processed | CSV consolidés (`df_full_spatial_corrected.csv`, `df_20000.csv`) et patchs normalisés (`patches_224x224_normalized`). |
| notebooks | Analyses exploratoires, pipeline SimpleCNN, grid search, études d'équité. |
| src | Modules réutilisables : chargement données (`data`), modèles (`models`), métriques (`evaluation`), utilitaires (`utils`). |
| models | Checkpoints (baseline, grid-search best, production), configs et métriques exportées. |
| results | Historique d'entraînement, rapports de métriques, prédictions patch/patient, figures. |
| documentation | Présents rapports techniques et méthodologiques. |

---

## 3. Données et prétraitements

1. Ingestion : fusion des métadonnées CAMELYON17 avec les chemins de patchs ; génération de CSV intermédiaires pour la reproductibilité.
2. Nettoyage : suppression des entrées invalides, harmonisation des identifiants patient, correction spatiale des coordonnées.
3. Séparation patient-stratifiée :
	- Entraînement/validation : centres 0, 1, 2 (StratifiedGroupKFold, 6 splits, seed fixe).
	- Test : centres 3, 4 pour simuler un hôpital jamais vu.
4. Transformations :
	- Train : flips horizontaux/verticaux, rotation 90°, ColorJitter, normalisation ImageNet.
	- Val/Test : normalisation uniquement.
5. Sampler : WeightedRandomSampler équilibrant virtuellement tumoraux et sains par classe.

---

## 4. Architecture et pipeline d'entraînement

- Backbone : ResNet-18 pré-entraîné ImageNet (finetuning complet).
- Tête personnalisée : linéaire 256 → ReLU → Dropout 0.4 → linéaire 1 (logit binaire).
- Perte : Focal Loss (gamma 2) avec coefficient alpha issu de la grille ; support BCE+`pos_weight` en fallback.
- Optimiseur : AdamW avec poids de décroissance réglés par grid search.
- Scheduler : ReduceLROnPlateau sur l'AUC validation (facteur 0.5, patience 2).
- Boucle : suivi loss/accuracy/F1/recall/AUC par époque, sauvegarde du meilleur état sur F1 validation, vidage mémoire GPU.

Notebooks principaux :
- Modèle 1 _ modelisation_SimpleCNN_patchs.ipynb : pipeline intégral (chargement, entraînement, interprétabilité).
- grid_search_simplecnn.ipynb : exploration initiale sur sous-échantillon de 20k patchs.
- grid_search_simplecnn_ResNet18.ipynb : grille finale sur lr/weight_decay/alpha.

---

## 5. Stratégie de recherche d'hyperparamètres

- Grille testée :
  - lr ∈ {1e-5, 5e-5, 1e-4}
  - weight_decay ∈ {0, 1e-4, 1e-3}
  - alpha (Focal Loss) ∈ {0.25, 0.5, 0.75}
  - epochs = 20
- Procédure :
  - Entraînement complet par combinaison avec suivi validation.
  - Historisation des métriques par époque et export CSV.
  - Sélection du meilleur checkpoint sur la F1 validation.
- Artefacts clés :
  - results/metrics/simplecnn_grid_search_results.csv (classement des runs).
  - models/final/simplecnn_grid_best.pth (poids retenus).
  - models/final/simplecnn_grid_best_metadata.json (hyperparamètres et métriques finales).

---

## 6. Évaluation et reporting

1. Patch-level : accuracy, F1 macro, recall, AUC ; matrice de confusion (Seaborn) et courbes ROC/PR (Plotly).
2. Patient-level : agrégation OR (positif si ≥1 patch tumoral), rapport classification sklearn, AUC moyenne des probabilités.
3. Analyse centre : récapitulatif métriques par hôpital (barplots) pour observer le domain shift.
4. Historique : results/training_history.csv pour tracer loss/metrics vs époques.
5. Exports :
	- results/predictions/test_predictions_patch_level.csv
	- results/predictions/test_predictions_patient_level.csv
	- results/final_metrics.json (synthèse patch/patient).

---

## 7. Interprétabilité et analyse d'erreurs

- Grad-CAM : calculé sur la dernière couche convolutionnelle (`backbone.layer4[-1]`), superposé aux patchs dans le notebook principal.
- Étude des faux négatifs/faux positifs : histogrammes de confiance, ventilation par centre, identification des échantillons difficiles.
- Restitution : figures prêtes à l'usage dans les rapports d'analyse clinique ou comités éthiques.

---

## 8. Artefacts de production

- Baseline : models/baseline_v1/best_model.pth (premier modèle stable).
- Releases : models/production/SimpleCNN_v1_<timestamp>.pth accompagnés de leurs fichiers *_config.json et *_metrics.json.
- Script d'inférence : models/production/inference.py (chargement checkpoint + pré/post-traitements).
- Documentation production : models/production/README.md indique la procédure d'utilisation.

---

## 9. Reproductibilité

1. Installer les dépendances via `uv pip install` ou `pip install -r` généré depuis pyproject.toml.
2. Vérifier la présence des patchs normalisés et CSV dans data/processed.
3. Exécuter les notebooks dans l'ordre :
	- EDA (EDA.ipynb, EDA_5000.ipynb) pour valider la cohérence des données.
	- Modèle SimpleCNN (Modèle 1 _ modelisation_SimpleCNN_patchs.ipynb).
	- Grid search (grid_search_simplecnn_*.ipynb).
4. Consulter results/ et models/ pour les artefacts générés.
5. Lancer la section Grad-CAM pour produire les visualisations interprétables.

---

## 10. Perspectives

- Étendre la recherche d'hyperparamètres (schedulers cosinus, warmup, gel progressif des blocs ResNet).
- Tester des architectures transformer (vision_transformer.ipynb) sous les mêmes splits patient.
- Automatiser la génération de rapports métriques + figures via scripts src/visualization et src/evaluation.
- Explorer des approches Multiple Instance Learning pour agréger les patchs au niveau patient.

---

Ce document fournit une vue d'ensemble du système. Chaque section renvoie vers des notebooks ou modules pour inspecter ou réexécuter les expériences détaillées.
