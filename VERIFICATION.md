# ✅ Vérification des fichiers requis

## Fichiers nécessaires

L'application nécessite deux fichiers dans le répertoire parent (`/Users/hectorchablis/Code/export/`) :

1. **analyze_image_metadata.py** - Script Python pour analyser les images
2. **metadata.csv** - Fichier CSV contenant les labels de cancer

## Vérification

Pour vérifier que les fichiers sont bien présents, exécutez :

```bash
cd /Users/hectorchablis/Code/export
ls -lh analyze_image_metadata.py metadata.csv
```

Vous devriez voir :
- `analyze_image_metadata.py` (environ 20 KB)
- `metadata.csv` (environ 15 MB)

## Structure des répertoires

```
export/
├── analyze_image_metadata.py    ← Script Python
├── metadata.csv                  ← Fichier CSV avec les labels
└── cancer-image-classifier/     ← Application Next.js
    ├── app/
    ├── components/
    └── ...
```

## Test de l'API

Pour tester que l'API peut accéder aux fichiers :

```bash
cd /Users/hectorchablis/Code/export/cancer-image-classifier
node -e "
const { resolve } = require('path');
const scriptPath = resolve(process.cwd(), '..', 'analyze_image_metadata.py');
const metadataPath = resolve(process.cwd(), '..', 'metadata.csv');
const fs = require('fs');
console.log('Script existe:', fs.existsSync(scriptPath), scriptPath);
console.log('Metadata existe:', fs.existsSync(metadataPath), metadataPath);
"
```

Les deux doivent retourner `true`.
