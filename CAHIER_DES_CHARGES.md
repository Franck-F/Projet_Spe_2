# Cahier des Charges Complet - Projet CAMELYON17

## 📋 Informations Officielles du Projet

### Contexte Médical

**Challenge** : CAMELYON17 (<https://camelyon17.grand-challenge.org/>)  
**Objectif clinique** : Développer des méthodes automatiques pour détecter les métastases dans les ganglions lymphatiques de patientes atteintes d'un cancer du sein.

**⚠️ IMPORTANT** : Vous ne construisez pas "juste un modèle", vous concevez un **système d'aide au diagnostic médical**.

### Objectifs Principaux

1. **Détection automatique** : Identifier la présence et l'étendue des métastases dans les lames histologiques
2. **Prédiction du stade pN** : Classifier chaque patiente selon le système pN (pN0, pN1, pN2, pN3)

---

## 🔬 Description du Dataset CAMELYON17

### 1. Nature des Données

**Whole Slide Images (WSI)** :

- Images issues de microscopes numériques
- Coloration H&E (Hématoxyline & Éosine)
- Très haute résolution (plusieurs gigapixels par lame)
- **⚠️ Les WSI ne peuvent pas être traitées directement par un CNN classique**

### 2. Découpage en Patchs

Pour rendre les données exploitables :

- **Taille standard** : 224 × 224 pixels
- **Format** : .png ou .tif
- **Type** : Images RGB
- Chaque patch = petite région locale du tissu

### 3. Labels au Niveau Patch

**Classification binaire ou multiclass** :

- `0` → Tissu normal
- `1` → Tissu tumoral

**Source** : Masque de vérité terrain fourni par des pathologistes experts

### 4. Labels au Niveau Patient (Stades pN)

**Classification clinique** indiquant :

- Le nombre de ganglions lymphatiques atteints
- L'étendue de la maladie

**Stades** :

- **pN0** : Aucun ganglion atteint
- **pN1** : Quelques ganglions positifs
- **pN2** : Atteinte modérée
- **pN3** : Atteinte étendue

---

## 🎯 Exigences Techniques

### Architecture du Système

```
WSI (Gigapixels)
    ↓
Découpage en patchs (224×224)
    ↓
Modèle CNN (Niveau Patch)
    ↓
Prédictions patch (0: normal, 1: tumoral)
    ↓
Agrégation (Niveau Patient)
    ↓
Prédiction stade pN (pN0, pN1, pN2, pN3)
```

### Métriques de Performance

**Niveau Patch** :

- Précision / Rappel
- F1-score
- AUC-ROC

**Niveau Patient** :

- Accuracy sur stades pN
- Cohen's Kappa
- Matrice de confusion 4×4

---

## 🤖 Intégration Module 4 : IA Responsable

### 1. Équité (Fairness)

#### Contexte Médical

En diagnostic médical, les algorithmes biaisés peuvent :

- Favoriser certains groupes démographiques
- Sous-représenter certaines ethnies
- Causer des soins inéquitables
- Générer des erreurs de diagnostic pour certains patients

#### Risques Spécifiques CAMELYON17

**Biais potentiels** :

- **Biais géographique** : 5 hôpitaux différents → variations de protocoles
- **Biais de coloration** : Différences de préparation des lames
- **Biais de population** : Sous-représentation de certains groupes
- **Biais de sévérité** : Déséquilibre entre stades pN

#### Stratégies de Correction

**1. Pré-processing** :

```python
# Rééquilibrage des données
from imblearn.over_sampling import SMOTE

# SMOTE pour équilibrer les classes pN
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_features, y_pn_stages)
```

**2. In-processing** :

```python
# Contrainte de fairness dans la loss
class FairLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, hospital_ids):
        # Loss principale
        ce = self.ce_loss(outputs, targets)
        
        # Pénalité de fairness (égalité entre hôpitaux)
        fairness_penalty = compute_hospital_disparity(outputs, hospital_ids)
        
        return ce + self.alpha * fairness_penalty
```

**3. Post-processing** :

```python
# Calibration par hôpital
def calibrate_by_hospital(predictions, hospital_ids):
    """
    Ajuste les seuils de décision pour chaque hôpital
    pour garantir l'équité des performances
    """
    calibrated_preds = []
    
    for hospital in unique_hospitals:
        mask = hospital_ids == hospital
        hospital_preds = predictions[mask]
        
        # Ajuster le seuil pour ce hôpital
        threshold = optimize_threshold(hospital_preds, targets[mask])
        calibrated = apply_threshold(hospital_preds, threshold)
        
        calibrated_preds.append(calibrated)
    
    return np.concatenate(calibrated_preds)
```

#### Métriques d'Équité à Calculer

**Demographic Parity** :

```python
# Proportion de prédictions positives par hôpital
for hospital in hospitals:
    positive_rate = (predictions[hospital_mask] == 1).mean()
    print(f"Hospital {hospital}: {positive_rate:.2%}")
```

**Equalized Odds** :

```python
# TPR et FPR par hôpital
from sklearn.metrics import confusion_matrix

for hospital in hospitals:
    cm = confusion_matrix(y_true[hospital_mask], y_pred[hospital_mask])
    tpr = cm[1,1] / (cm[1,1] + cm[1,0])  # Sensibilité
    fpr = cm[0,1] / (cm[0,1] + cm[0,0])  # 1 - Spécificité
    print(f"Hospital {hospital}: TPR={tpr:.3f}, FPR={fpr:.3f}")
```

### 2. Transparence / Explicabilité

#### Pourquoi c'est Critique en Médical

Un pathologiste doit pouvoir :

- **Comprendre** pourquoi le modèle a fait une prédiction
- **Valider** que le modèle regarde les bonnes structures
- **Corriger** si le modèle se trompe
- **Faire confiance** au système

#### Outils à Implémenter

**1. SHAP (SHapley Additive exPlanations)** :

```python
import shap

# Niveau agrégation (features extraites)
explainer = shap.TreeExplainer(xgboost_model)
shap_values = explainer.shap_values(X_patient_features)

# Visualiser
shap.summary_plot(shap_values, X_patient_features, 
                  feature_names=['tumor_percentage', 'max_prob', 'std_prob', ...])
```

**2. Grad-CAM (Pour CNN)** :

```python
from src.evaluation.interpretability import GradCAM

# Identifier les régions importantes
gradcam = GradCAM(model, target_layer)
heatmap = gradcam.generate_cam(input_patch, target_class=1)

# Visualiser
plot_gradcam_heatmap(original_patch, heatmap)
```

**3. Feature Importance** :

```python
# Pour le modèle d'agrégation
import plotly.express as px

feature_importance = xgb_model.feature_importances_
features = ['tumor_pct', 'mean_prob', 'max_prob', 'std_prob', ...]

fig = px.bar(x=features, y=feature_importance, 
             title='Importance des Features pour Prédiction pN')
fig.show()
```

### 3. Monitoring et Drift Detection

#### Types de Drift à Surveiller

**1. Feature Drift (Covariate Drift)** :

```python
from scipy.stats import ks_2samp

# Comparer distribution des features entre train et production
def detect_feature_drift(X_train, X_prod, threshold=0.05):
    drift_detected = {}
    
    for col in X_train.columns:
        statistic, p_value = ks_2samp(X_train[col], X_prod[col])
        drift_detected[col] = p_value < threshold
    
    return drift_detected
```

**2. Concept Drift** :

```python
# Surveiller les performances dans le temps
def monitor_performance(model, data_stream):
    window_size = 1000
    performance_history = []
    
    for batch in data_stream:
        y_pred = model.predict(batch['X'])
        accuracy = (y_pred == batch['y']).mean()
        performance_history.append(accuracy)
        
        # Alerte si dégradation
        if len(performance_history) > 10:
            recent_perf = np.mean(performance_history[-10:])
            baseline_perf = np.mean(performance_history[:10])
            
            if recent_perf < baseline_perf - 0.05:
                print("⚠️ DRIFT DÉTECTÉ : Réentraînement recommandé")
```

**3. Target Drift** :

```python
# Surveiller la distribution des prédictions
def monitor_prediction_distribution(predictions, baseline_dist):
    current_dist = np.bincount(predictions) / len(predictions)
    
    # KL divergence
    kl_div = np.sum(current_dist * np.log(current_dist / baseline_dist))
    
    if kl_div > threshold:
        print("⚠️ Distribution des prédictions a changé")
```

### 4. Traduire Métriques Techniques → Indicateurs Business

#### ROI (Retour sur Investissement)

**Calcul** :

```
Coûts évités = (Nombre de FN évités) × (Coût d'un traitement retardé)
Coûts du système = Développement + Déploiement + Maintenance

ROI = (Coûts évités - Coûts du système) / Coûts du système × 100%
```

**Exemple** :

```python
# Hypothèses
fn_rate_manual = 0.05  # 5% de faux négatifs en manuel
fn_rate_ai = 0.01      # 1% avec IA
cost_per_fn = 50000    # Coût d'un traitement retardé (€)
n_patients_year = 10000

# Calcul
fn_avoided = (fn_rate_manual - fn_rate_ai) * n_patients_year
cost_avoided = fn_avoided * cost_per_fn

print(f"Faux négatifs évités : {fn_avoided:.0f} patients/an")
print(f"Coûts évités : {cost_avoided:,.0f} €/an")
```

#### KPI (Key Performance Indicators)

| Métrique Technique | KPI Business | Impact Clinique |
|-------------------|--------------|-----------------|
| **Recall > 95%** | Moins de 5% de métastases manquées | Patients reçoivent traitement approprié |
| **Precision > 90%** | Moins de 10% de faux positifs | Évite biopsies inutiles |
| **Temps de traitement** | < 5 min par patient | Accélère le diagnostic |
| **Drift < 5%** | Performances stables | Fiabilité à long terme |

---

## 📊 Livrables Attendus

### 1. Code et Notebooks

**Structure** :

```
notebooks/
├── 01_EDA.ipynb                    # Analyse exploratoire
├── 02_preprocessing.ipynb          # Prétraitement
├── 03_modeling_patch.ipynb         # Modélisation CNN
├── 04_aggregation.ipynb            # Agrégation patient
├── 05_evaluation.ipynb             # Évaluation complète
└── 06_fairness_transparency.ipynb  # ⭐ NOUVEAU : IA responsable
```

### 2. Rapport Final (Max 15 pages)

**Structure imposée** :

1. **Introduction** (1 page)
   - Contexte médical
   - Enjeux cliniques
   - Objectifs

2. **Données et Prétraitement** (2 pages)
   - Description CAMELYON17
   - Stratégie de sous-échantillonnage
   - Pipeline de preprocessing

3. **Méthodologie** (4 pages)
   - Architecture CNN
   - Stratégie d'agrégation
   - Gestion du déséquilibre

4. **Résultats** (4 pages)
   - Performances (patch + patient)
   - Comparaison des approches
   - Visualisations

5. **⭐ IA Responsable** (3 pages) - **NOUVEAU**
   - **Analyse d'équité** :
     - Performances par hôpital
     - Métriques de fairness
     - Stratégies de correction
   - **Transparence** :
     - SHAP / Grad-CAM
     - Interprétation des décisions
   - **Monitoring** :
     - Stratégie de drift detection
     - Plan de maintenance

6. **Discussion et Limites** (1 page)
   - Biais identifiés
   - Recommandations cliniques
   - Travaux futurs

### 3. Présentation Orale (15 min)

**Slides obligatoires** :

1. Contexte médical et enjeux
2. Dataset et défis
3. Architecture du système
4. Résultats (performances)
5. **⭐ Équité et transparence** (2-3 slides)
6. Recommandations et conclusion

---

## 🔧 Outils et Bibliothèques Recommandés

### IA Responsable

```toml
# À ajouter dans pyproject.toml

[dependency-groups]
fairness = [
    "fairlearn>=0.8.0",      # Métriques et mitigation de biais
    "aif360>=0.5.0",         # AI Fairness 360 (IBM)
]

interpretability = [
    "shap>=0.42.0",          # SHAP values
    "lime>=0.2.0",           # LIME
    "grad-cam>=1.4.8",       # Grad-CAM
]

monitoring = [
    "evidently>=0.4.0",      # Drift detection
    "alibi-detect>=0.11.0",  # Outlier & drift detection
]
```

### Installation

```bash
# Installer les groupes de dépendances
uv sync --group fairness
uv sync --group interpretability
uv sync --group monitoring
```

---

## ✅ Checklist de Conformité

### Exigences Projet

- [ ] Détection automatique de métastases (niveau patch)
- [ ] Prédiction stade pN (niveau patient)
- [ ] Gestion des WSI gigapixels (découpage en patchs)
- [ ] Labels binaires au niveau patch (0: normal, 1: tumoral)
- [ ] Labels multiclass au niveau patient (pN0-pN3)

### Exigences Module 4 : IA Responsable

**Équité** :

- [ ] Analyse des biais potentiels (géographiques, démographiques)
- [ ] Métriques de fairness calculées (demographic parity, equalized odds)
- [ ] Stratégie de correction implémentée (SMOTE, weighted loss, calibration)
- [ ] Performances par sous-groupe documentées

**Transparence** :

- [ ] SHAP implémenté (niveau agrégation)
- [ ] Grad-CAM implémenté (niveau patch)
- [ ] Feature importance visualisée
- [ ] Interprétation médicale des décisions

**Monitoring** :

- [ ] Stratégie de drift detection définie
- [ ] Métriques de monitoring identifiées
- [ ] Plan de maintenance documenté

**Business** :

- [ ] ROI calculé et justifié
- [ ] KPI business définis
- [ ] Impact clinique quantifié

---

## 🎯 Critères d'Évaluation (Estimés)

| Critère | Poids | Description |
|---------|-------|-------------|
| **Performances techniques** | 30% | Recall, Precision, Accuracy (patch + patient) |
| **Méthodologie** | 25% | Rigueur scientifique, choix justifiés |
| **IA Responsable** | 25% | Équité, transparence, monitoring |
| **Documentation** | 10% | Clarté du rapport et du code |
| **Présentation** | 10% | Qualité de la communication orale |

---

## 📚 Ressources Complémentaires

### Fairness & Bias

- Bias and fairness in ML : <https://timkimutai.medium.com/bias-and-fairness-in-machine-learning-a-beginners-guide-to-building-models-that-don-t-play-c9a503c3c78b>
- COMPAS case study : <http://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm>
- AI in healthcare bias : <https://jamanetwork.com/journals/jama/article-abstract/2823006>

### Explainability

- SHAP paper : <https://arxiv.org/abs/1705.07874>
- LIME paper : <https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf>
- Grad-CAM paper : <https://arxiv.org/abs/1610.02391>

### Neural Networks

- CNN tutorial PyTorch : <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>
- Neural networks survey : <https://arxiv.org/abs/2305.17473>

---

**Ce document constitue votre cahier des charges complet. Assurez-vous de couvrir TOUS les aspects, en particulier l'IA responsable qui est au cœur du Module 4 ! 🚀**
