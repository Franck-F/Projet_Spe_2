# Glossaire Médical - Projet CAMELYON17

## Contexte Clinique

### Cancer du Sein

Le cancer du sein est une tumeur maligne qui se développe dans les cellules mammaires. C'est le cancer le plus fréquent chez la femme.

### Métastases Ganglionnaires

- **Définition** : Propagation de cellules cancéreuses du sein vers les ganglions lymphatiques axillaires (sous l'aisselle)
- **Importance** : Indicateur majeur du stade du cancer et du pronostic
- **Impact** : Détermine le traitement (chimiothérapie, radiothérapie)

## Système de Classification pN

Le système **pN** (pathological Node) classifie l'atteinte ganglionnaire après examen histopathologique :

### pN0 - Pas de métastase

- Aucune cellule cancéreuse détectée dans les ganglions
- Meilleur pronostic

### pN1 - Métastase limitée

- 1 à 3 ganglions axillaires atteints
- OU micro-métastases (≤ 2mm)
- OU cellules tumorales isolées (≤ 0.2mm)

### pN2 - Métastase modérée

- 4 à 9 ganglions axillaires atteints
- OU ganglions mammaires internes atteints (sans ganglions axillaires)

### pN3 - Métastase étendue

- ≥ 10 ganglions axillaires atteints
- OU ganglions sous-claviculaires atteints
- OU ganglions mammaires internes + axillaires atteints
- Pronostic plus réservé

## Terminologie Histopathologique

### Whole Slide Image (WSI)

- **Définition** : Image numérique haute résolution d'une lame histologique complète
- **Caractéristiques** : Très grande taille (plusieurs Go), multi-résolution
- **Format** : Pyramide d'images à différents niveaux de zoom

### Coloration H&E (Hématoxyline et Éosine)

- **Hématoxyline** : Colore les noyaux cellulaires en bleu/violet
- **Éosine** : Colore le cytoplasme et structures extracellulaires en rose/rouge
- **Standard** : Coloration la plus utilisée en anatomopathologie

### Patch

- **Définition** : Petite région extraite d'une WSI (ex: 224x224 pixels)
- **Utilité** : Permet l'analyse par deep learning (WSI trop grandes)

### Ganglion Lymphatique

- **Fonction** : Organe du système immunitaire filtrant la lymphe
- **Structure** : Cortex (périphérie) et médulla (centre)
- **Pathologie** : Les métastases se développent généralement dans le cortex

## Enjeux Cliniques

### Faux Négatif (FN)

- **Définition** : Métastase présente mais non détectée
- **Conséquence** : **CRITIQUE** - Sous-traitement du patient, risque de récidive
- **Priorité** : Minimiser les FN → **Maximiser le Recall**

### Faux Positif (FP)

- **Définition** : Métastase détectée à tort
- **Conséquence** : Sur-traitement (chimiothérapie inutile, effets secondaires)
- **Impact** : Moins grave que FN, mais à limiter

### Micro-métastases

- **Définition** : Amas de cellules tumorales de 0.2mm à 2mm
- **Défi** : Difficiles à détecter, même pour les pathologistes
- **Importance** : Peuvent influencer le traitement

## Dataset CAMELYON17

### Origine

- **Challenge** : CAMELYON17 (Cancer Metastases in Lymph Nodes)
- **Année** : 2017
- **Centres** : 5 hôpitaux différents (variabilité inter-hospitalière)

### Variabilité

- **Protocoles de coloration** : Différences entre hôpitaux
- **Scanners** : Différents équipements de numérisation
- **Domain Shift** : Défi majeur pour la généralisation du modèle

## Workflow Clinique

### Processus Actuel

1. **Prélèvement** : Biopsie ou exérèse du ganglion
2. **Préparation** : Fixation, inclusion en paraffine, coupe fine
3. **Coloration** : H&E
4. **Numérisation** : Scanner de lames → WSI
5. **Diagnostic** : Examen par pathologiste (peut prendre plusieurs heures)

### Apport de l'IA

- **Pré-screening** : Identifier les lames suspectes en priorité
- **Second avis** : Aide à la décision pour cas difficiles
- **Gain de temps** : Accélérer le workflow diagnostique
- **Standardisation** : Réduire la variabilité inter-observateur

## Références Médicales

- **TNM Classification** : Système international de stadification des cancers
- **AJCC** : American Joint Committee on Cancer
- **WHO** : World Health Organization - Classification des tumeurs

---

**Note** : Ce glossaire sera enrichi au fur et à mesure du projet avec les termes spécifiques rencontrés.
