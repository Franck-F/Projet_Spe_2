# Documentation Technique : Modélisation CNN pour la Détection de Métastases

Ce document détaille chaque étape du notebook `notebooks/modelisation_SimpleCNN_patchs.ipynb`. Il explique les choix techniques, les concepts scientifiques et l'importance de chaque bloc de code dans le cadre du projet de détection de métastases de cancer du sein (Challenge CAMELYON17).

---

## 1. Introduction et Objectifs

Le notebook a pour but d'entraîner un réseau de neurones convolutif (CNN) capable d'identifier si un "patch" (petite image extraite d'une lame histologique) contient des cellules tumorales ou non.

**Points clés :**

* **Dataset** : Échantillon de 20 000 patchs (redimensionnés en 224x224).
* **Split stratégique** : Séparation par centre hospitalier pour tester la robustesse du modèle face à des données provenant de sources différentes (Domain Shift).

---

## 2. Configuration et Environnement

**Pourquoi ?** Initialiser les bibliothèques indispensables (PyTorch pour le Deep Learning, Pandas pour les données, Matplotlib/Plotly pour les graphiques).
**Concepts clés :**

* **DEVICE (CPU/CUDA)** : Détermine si les calculs se font sur le processeur (lent) ou la carte graphique (rapide).
* **BATCH_SIZE (64)** : Nombre d'images traitées simultanément par le modèle. Un batch trop grand sature la mémoire, un batch trop petit rend l'entraînement instable.
* **LEARNING RATE (LR (1e-4) assure que le modèle apprend sûrement et sans s'éparpiller.)** : La vitesse à laquelle le modèle ajuste ses "connaissances". S'il est trop haut, le modèle diverge ; s'il est trop bas, il n'apprend rien.
* **Le Weight Decay (1e-2)** : assure que le modèle reste simple et robuste, pour être capable de détecter le cancer même sur des images provenant d'un nouveau scanner

---

## 3. Analyse Exploratoire des Données (EDA)

**Pourquoi ?** Comprendre la répartition des données avant de lancer l'entraînement.
**Importance :**

* **Déséquilibre des classes** : On observe que ~70% des patchs sont sains (0) contre ~30% tumoraux (1). Si on ne traite pas ce déséquilibre, le modèle risque d'être "fainéant" et de prédire "sain" tout le temps pour avoir 70% de précision.
* **Distribution par Centre** : On vérifie que chaque hôpital a un ratio tumeur/sain cohérent pour éviter des biais de centre.

---

## 4. Visualisation des Échantillons

**Pourquoi ?** Une vérification "de bon sens".
**Importance :** Cela confirme que les images sont bien chargées, que les labels correspondent visuellement aux structures tissulaires et que la normalisation des couleurs n'a pas détruit l'information médicale.

---

## 5. Stratégie de Split (Train/Val/Test)

**Pourquoi ?** C'est l'étape la plus critique du challenge WILDS/CAMELYON17.
**Concept : Domain Shift**

* On entraîne sur les **Centres 0, 1 et 2**.
* On valide sur une partie de ces centres.
* On teste sur les **Centres 3 et 4** (données totalement inconnues).
**Importance :** Tester sur des centres différents simule la réalité clinique : une IA entraînée dans un hôpital A doit fonctionner dans un hôpital B, même si les scanners ou les protocoles de coloration diffèrent.

---

## 6. Gestion du Déséquilibre (Sampler)

**Pourquoi ?** Corriger le ratio 70/30 vu précédemment.
**Concept : WeightedRandomSampler**
On donne un "poids" plus élevé aux images tumorales lors de la sélection pour que, durant une époque d'entraînement, le modèle voie autant d'images saines que d'images malades (ratio 50/50 "virtuel").

---

## 7. Architecture du Modèle (Simple CNN)

**Pourquoi ?** Créer le "cerveau" capable d'extraire des caractéristiques visuelles.
**Composants détaillés :**

1. **Convolutions (nn.Conv2d)** : Des filtres qui détectent des formes (bords, noyaux cellulaires, textures).
2. **BatchNorm** : Stabilise l'entraînement en recentrant les données entre chaque couche.
3. **ReLU** : Fonction d'activation qui permet de modéliser des relations complexes (non-linéaires).
4. **MaxPool** : Réduit la taille de l'image pour ne garder que les informations les plus importantes (gain de vitesse).
5. **Dropout** : Désactive aléatoirement des neurones pour forcer le modèle à ne pas trop apprendre par cœur (évite le Surapprentissage / Overfitting).
6. **Fully Connected (Dense)** : Couches finales qui prennent toutes les formes détectées pour décider : "Probabilité Tumeur".

---

## 8. Entraînement (Training Loop)

**Pourquoi ?** La phase où le modèle apprend par essai/erreur.
**Importance :**

* **Binary Cross Entropy (BCE)** : La "punition" reçue par le modèle quand il se trompe.
* **Adam Optimizer** : L'algorithme qui ajuste les poids du modèle pour minimiser la punition.
* **Scheduler** : Réduit automatiquement la vitesse d'apprentissage (LR) au fil du temps pour affiner les réglages finaux.

---

## 9. Évaluation et Métriques

**Pourquoi ?** Mesurer la fiabilité du système.
**Définitions :**

* **Précision** : "Sur tout ce que j'ai dit malade, combien le sont vraiment ?" (Évite les faux positifs).
* **Recall (Rappel)** : "Sur tous les malades réels, combien en ai-je trouvés ?" (Évite les faux négatifs - crucial en médecine).
* **F1-Score** : Moyenne harmonieuse entre Précision et Rappel.
* **AUC (Area Under Curve)** : Score global de performance (1.0 = parfait). Elle mesure la capacité du modèle à classer un cas malade au-dessus d'un cas sain.

---

## 10. Conclusion du Notebook

Ce pipeline transforme des données brutes en un prédicteur robuste. En se concentrant sur le **Généralisation Inter-Hôpitaux**, il garantit que le modèle n'apprend pas juste des caractéristiques spécifiques à un scanner, mais bien la signature visuelle du cancer.
