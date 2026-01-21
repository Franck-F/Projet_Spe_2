#!/usr/bin/env node
/**
 * Script de test pour vérifier que les chemins vers les fichiers Python et CSV sont corrects
 */

const { resolve } = require('path');
const { existsSync } = require('fs');

const scriptPath = resolve(__dirname, '..', 'analyze_image_metadata.py');
const metadataPath = resolve(__dirname, '..', 'metadata.csv');

console.log('🔍 Vérification des chemins...\n');
console.log('Répertoire courant:', __dirname);
console.log('Répertoire parent:', resolve(__dirname, '..'));
console.log();

console.log('📄 Script Python:');
console.log('  Chemin:', scriptPath);
console.log('  Existe:', existsSync(scriptPath) ? '✅ OUI' : '❌ NON');
console.log();

console.log('📊 Fichier metadata.csv:');
console.log('  Chemin:', metadataPath);
console.log('  Existe:', existsSync(metadataPath) ? '✅ OUI' : '❌ NON');
console.log();

if (existsSync(scriptPath) && existsSync(metadataPath)) {
  console.log('✅ Tous les fichiers sont présents !');
  console.log('🚀 Vous pouvez démarrer l\'application avec: npm run dev');
} else {
  console.log('❌ Certains fichiers manquent !');
  console.log('   Assurez-vous que les fichiers sont dans le répertoire parent.');
}
