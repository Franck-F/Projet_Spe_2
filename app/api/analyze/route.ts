import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink, access, constants } from 'fs/promises';
import { join, resolve, basename } from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Chemins vers les fichiers Python et CSV (dans le répertoire parent)
const getScriptPath = () => {
  // Depuis cancer-image-classifier, remonter d'un niveau pour trouver les fichiers
  return resolve(process.cwd(), '..', 'analyze_image_metadata.py');
};

const getMetadataPath = () => {
  return resolve(process.cwd(), '..', 'metadata.csv');
};

export async function POST(request: NextRequest) {
  console.log('📥 Requête POST reçue sur /api/analyze');
  
  try {
    const formData = await request.formData();
    const file = formData.get('image') as File;
    const useMetadataParam = formData.get('useMetadata');
    const useMetadata = useMetadataParam === 'true' || useMetadataParam === null; // Par défaut true

    console.log('📁 Fichier reçu:', file ? {
      name: file.name,
      type: file.type,
      size: file.size
    } : 'Aucun fichier');
    console.log('📊 Utiliser metadata.csv:', useMetadata);

    if (!file) {
      console.error('❌ Aucune image fournie');
      return NextResponse.json(
        { error: 'Aucune image fournie' },
        { status: 400 }
      );
    }

    // Vérifier que le script Python existe
    const scriptPath = getScriptPath();
    const metadataPath = getMetadataPath();

    console.log('🔍 Vérification des chemins:');
    console.log('  Script:', scriptPath);
    console.log('  Metadata:', metadataPath);

    try {
      await access(scriptPath, constants.F_OK);
      console.log('✅ Script Python trouvé');
    } catch {
      console.error('❌ Script Python non trouvé:', scriptPath);
      return NextResponse.json(
        { error: `Script Python non trouvé: ${scriptPath}` },
        { status: 500 }
      );
    }

    // Vérifier metadata.csv seulement si on doit l'utiliser
    if (useMetadata) {
      try {
        await access(metadataPath, constants.F_OK);
        console.log('✅ Metadata CSV trouvé');
      } catch {
        console.error('❌ Metadata CSV non trouvé:', metadataPath);
        return NextResponse.json(
          { error: `Fichier metadata.csv non trouvé: ${metadataPath}` },
          { status: 500 }
        );
      }
    } else {
      console.log('⚠️ Metadata CSV ignoré (mode sans labels)');
    }

    // Sauvegarder temporairement le fichier
    // Préserver le nom de fichier original pour l'extraction des métadonnées
    console.log('💾 Sauvegarde du fichier temporaire...');
    console.log('📝 Nom de fichier original:', file.name);
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    // Utiliser le nom de fichier original si possible, sinon ajouter un préfixe
    const sanitizedName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
    const tempPath = join('/tmp', sanitizedName.startsWith('patch_') ? sanitizedName : `upload_${Date.now()}_${sanitizedName}`);
    await writeFile(tempPath, buffer);
    console.log('✅ Fichier sauvegardé:', tempPath);
    console.log('📝 Nom de fichier utilisé pour extraction:', basename(tempPath));

    try {
      // Appeler le script Python pour analyser l'image
      // Si useMetadata est false, ne pas passer --metadata
      const command = useMetadata
        ? `python3 "${scriptPath}" "${tempPath}" --metadata "${metadataPath}" --json-only`
        : `python3 "${scriptPath}" "${tempPath}" --json-only --no-metadata`;
      console.log('🐍 Exécution de la commande Python:', command);
      
      const { stdout, stderr } = await execAsync(command, {
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer
        timeout: 60000 // 60 secondes timeout
      });

      console.log('📊 Sortie Python (stdout, premiers 500 caractères):', stdout.substring(0, 500));

      // Filtrer les warnings matplotlib qui ne sont pas des erreurs
      if (stderr && !stderr.includes('Matplotlib') && !stderr.includes('font cache')) {
        console.warn('Avertissement Python:', stderr);
      }

      // Parser la sortie JSON du script Python
      // Le script Python affiche le JSON à la fin avec --json-only
      const jsonMatch = stdout.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        console.error('❌ Impossible de trouver le JSON dans la sortie');
        console.error('Sortie complète:', stdout);
        throw new Error('Impossible de parser les résultats JSON depuis la sortie Python');
      }

      let results;
      try {
        results = JSON.parse(jsonMatch[0]);
        console.log('✅ JSON parsé avec succès');
      } catch (parseError: any) {
        console.error('❌ Erreur de parsing JSON:', parseError);
        console.error('JSON brut (premiers 1000 caractères):', jsonMatch[0].substring(0, 1000));
        throw new Error(`Erreur lors du parsing du JSON: ${parseError.message}`);
      }

      // Nettoyer le fichier temporaire
      await unlink(tempPath);
      console.log('✅ Fichier temporaire supprimé');

      return NextResponse.json(results);
    } catch (error: any) {
      // Nettoyer le fichier temporaire en cas d'erreur
      try {
        await unlink(tempPath);
        console.log('🧹 Fichier temporaire nettoyé après erreur');
      } catch {}

      console.error('❌ Erreur lors de l\'analyse:', error);
      console.error('Stack:', error.stack);
      return NextResponse.json(
        { error: `Erreur lors de l'analyse: ${error.message || error}` },
        { status: 500 }
      );
    }
  } catch (error: any) {
    console.error('❌ Erreur serveur:', error);
    console.error('Stack:', error.stack);
    return NextResponse.json(
      { error: `Erreur serveur: ${error.message || error}` },
      { status: 500 }
    );
  }
}
