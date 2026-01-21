# 🔍 Guide de débogage - Upload d'image

## Étapes pour déboguer

1. **Ouvrez la console du navigateur** (F12 → Console)

2. **Cliquez sur "Sélectionner une image"**

3. **Vérifiez les logs dans la console** :
   - Vous devriez voir : `🔘 Bouton cliqué`
   - Puis : `📂 handleFileInput appelé`
   - Puis : `✅ Fichier trouvé dans input: [nom du fichier]`
   - Puis : `📁 handleFile appelé avec: {...}`
   - Puis : `✅ Preview créé`
   - Puis : `🚀 Démarrage de l'analyse...`
   - Puis : `📤 Envoi de la requête à /api/analyze...`

4. **Vérifiez les logs du serveur** (terminal où `npm run dev` tourne) :
   - Vous devriez voir : `📥 Requête POST reçue sur /api/analyze`
   - Puis : `📁 Fichier reçu: {...}`
   - Puis : `🔍 Vérification des chemins:`
   - Puis : `✅ Script Python trouvé`
   - Puis : `✅ Metadata CSV trouvé`
   - Puis : `💾 Sauvegarde du fichier temporaire...`
   - Puis : `🐍 Exécution de la commande Python: ...`

## Problèmes courants

### Le bouton ne fait rien
- Vérifiez que `fileInputRef.current` n'est pas null dans les logs
- Vérifiez qu'il n'y a pas d'erreurs JavaScript dans la console

### Le fichier est sélectionné mais rien ne se passe
- Vérifiez que `handleFileInput` est appelé
- Vérifiez que le type de fichier commence par `image/`
- Vérifiez qu'il n'y a pas d'erreurs dans `handleFile`

### Le preview ne s'affiche pas
- Vérifiez que `setPreview` est appelé
- Vérifiez qu'il n'y a pas d'erreurs dans le FileReader
- Vérifiez la console pour les erreurs de chargement d'image

### L'API ne répond pas
- Vérifiez les logs du serveur
- Vérifiez que les chemins Python/CSV sont corrects
- Vérifiez qu'il n'y a pas d'erreurs dans l'exécution Python
