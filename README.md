# 🔬 Cancer Image Classifier

Application Next.js pour classifier les images de patches histologiques et détecter la présence de cancer.

## 🚀 Fonctionnalités

- **Upload d'images** : Interface drag-and-drop pour uploader des images
- **Analyse automatique** : Détection du label de cancer (0 = pas de cancer, 1 = cancer)
- **Affichage des résultats** : Métadonnées complètes avec statistiques des pixels
- **Interface moderne** : Design responsive avec Tailwind CSS

## 📋 Prérequis

- Node.js 18+ 
- Python 3.9+
- Les dépendances Python : `matplotlib`, `numpy`, `Pillow`
- Le fichier `metadata.csv` dans le répertoire parent
- Le script `analyze_image_metadata.py` dans le répertoire parent

## 🛠️ Installation

1. Installer les dépendances Python :
```bash
pip3 install matplotlib numpy Pillow
```

2. Installer les dépendances Next.js :
```bash
cd cancer-image-classifier
npm install
```

## ▶️ Démarrage

1. Démarrer le serveur de développement :
```bash
npm run dev
```

2. Ouvrir [http://localhost:3000](http://localhost:3000) dans votre navigateur

## 📁 Structure du projet

```
cancer-image-classifier/
├── app/
│   ├── api/
│   │   └── analyze/
│   │       └── route.ts      # API route pour analyser les images
│   └── page.tsx              # Page principale
├── components/
│   ├── ImageUpload.tsx       # Composant d'upload d'image
│   └── ResultsDisplay.tsx    # Composant d'affichage des résultats
└── README.md
```

## 🔧 Configuration

Si vos fichiers Python sont dans un autre emplacement, modifiez les chemins dans `app/api/analyze/route.ts` :

```typescript
const scriptPath = join(process.cwd(), '..', 'analyze_image_metadata.py');
const metadataPath = join(process.cwd(), '..', 'metadata.csv');
```

## 📝 Utilisation

1. Ouvrez l'application dans votre navigateur
2. Glissez-déposez une image ou cliquez pour sélectionner
3. L'image sera analysée automatiquement
4. Les résultats s'afficheront avec :
   - Label de cancer (0 ou 1)
   - Métadonnées complètes
   - Statistiques des pixels par canal

## 🎨 Format des images

L'application fonctionne avec les images au format Camelyon17 :
- Format : PNG
- Dimensions : 96×96 pixels (ou autres)
- Nom de fichier : `patch_patient_XXX_node_X_x_XXXX_y_XXXX.png`

## 🐛 Dépannage

- **Erreur "Python script not found"** : Vérifiez que le chemin vers `analyze_image_metadata.py` est correct
- **Erreur "metadata.csv not found"** : Assurez-vous que le fichier `metadata.csv` est dans le répertoire parent
- **Erreur d'upload** : Vérifiez que le dossier `/tmp` est accessible en écriture

## 📄 Licence

ISC
