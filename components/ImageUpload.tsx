'use client';

import { useState, useRef } from 'react';

interface ImageUploadProps {
  onUploadStart: () => void;
  onAnalysisComplete: (data: any) => void;
}

export default function ImageUpload({ onUploadStart, onAnalysisComplete }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [useMetadata, setUseMetadata] = useState(true); // Par défaut, utiliser metadata.csv
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log('📂 handleFileInput appelé');
    console.log('Fichiers:', e.target.files);
    if (e.target.files && e.target.files[0]) {
      console.log('✅ Fichier trouvé dans input:', e.target.files[0].name);
      handleFile(e.target.files[0]);
    } else {
      console.warn('⚠️ Aucun fichier dans l\'input');
    }
  };

  const handleFile = async (file: File) => {
    console.log('📁 handleFile appelé avec:', {
      name: file.name,
      type: file.type,
      size: file.size,
      lastModified: file.lastModified
    });
    
    // Vérifier que c'est une image
    if (!file.type.startsWith('image/')) {
      console.warn('⚠️ Ce n\'est pas une image:', file.type);
      alert('Veuillez uploader une image (format: ' + file.type + ')');
      return;
    }

    console.log('✅ Type d\'image valide, création du preview...');
    
    // Créer un aperçu immédiatement
    const reader = new FileReader();
    reader.onload = (e) => {
      console.log('✅ Preview créé, longueur:', (e.target?.result as string)?.length);
      setPreview(e.target?.result as string);
    };
    reader.onerror = (error) => {
      console.error('❌ Erreur lors de la lecture du fichier:', error);
      alert('Erreur lors de la lecture de l\'image');
    };
    reader.onloadstart = () => {
      console.log('🔄 Début de la lecture du fichier...');
    };
    reader.readAsDataURL(file);

    // Uploader et analyser
    console.log('🚀 Démarrage de l\'analyse...');
    onUploadStart();
    const formData = new FormData();
    formData.append('image', file);

    try {
      console.log('📤 Envoi de la requête à /api/analyze...');
      console.log('📊 Utiliser metadata.csv:', useMetadata);
      
      // Ajouter le paramètre useMetadata au FormData
      formData.append('useMetadata', useMetadata ? 'true' : 'false');
      
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      console.log('📥 Réponse reçue:', response.status, response.statusText);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Erreur inconnue' }));
        console.error('❌ Erreur de réponse:', errorData);
        throw new Error(errorData.error || `Erreur HTTP: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ Données reçues:', data);
      onAnalysisComplete(data);
    } catch (error: any) {
      console.error('❌ Erreur lors de l\'analyse:', error);
      alert(`Erreur lors de l'analyse de l'image: ${error.message || error}`);
      onAnalysisComplete(null);
    }
  };

  return (
    <div>
      {/* Toggle pour utiliser metadata.csv - Gros bouton */}
      <div className="mb-6">
        <button
          type="button"
          onClick={() => {
            setUseMetadata(!useMetadata);
            console.log('📊 Utiliser metadata.csv:', !useMetadata);
          }}
          className={`w-full py-4 px-6 rounded-lg border-2 transition-all duration-200 ${
            useMetadata
              ? 'bg-indigo-600 border-indigo-700 text-white shadow-lg hover:bg-indigo-700'
              : 'bg-gray-100 border-gray-300 text-gray-700 hover:bg-gray-200'
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
                useMetadata
                  ? 'bg-white border-white'
                  : 'bg-white border-gray-400'
              }`}>
                {useMetadata && (
                  <svg className="w-4 h-4 text-indigo-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </div>
              <div className="text-left">
                <div className="font-bold text-lg">
                  {useMetadata ? '✓ Détection de cancer activée' : '⚠️ Détection de cancer désactivée'}
                </div>
                <div className={`text-sm mt-1 ${
                  useMetadata ? 'text-indigo-100' : 'text-gray-600'
                }`}>
                  {useMetadata 
                    ? 'L\'analyse utilisera les labels du fichier metadata.csv pour détecter le cancer'
                    : 'L\'analyse ne pourra pas détecter le cancer (seulement les métadonnées techniques)'}
                </div>
              </div>
            </div>
            <div className={`text-2xl ${useMetadata ? 'text-white' : 'text-gray-400'}`}>
              {useMetadata ? '🔬' : '📊'}
            </div>
          </div>
        </button>
      </div>

      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-indigo-500 bg-indigo-50'
            : 'border-gray-300 hover:border-indigo-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          id="file-upload"
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
        />

        {preview ? (
          <div className="space-y-4">
            <div className="relative">
              <img
                src={preview}
                alt="Preview"
                className="max-w-full max-h-64 mx-auto rounded-lg shadow-md"
                onLoad={() => console.log('✅ Image preview chargée')}
                onError={(e) => {
                  console.error('❌ Erreur de chargement de l\'image preview:', e);
                  setPreview(null);
                }}
              />
            </div>
            <button
              type="button"
              onClick={() => {
                console.log('🔄 Changement d\'image demandé');
                setPreview(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              className="text-sm text-gray-600 hover:text-gray-800 underline"
            >
              Changer d'image
            </button>
          </div>
        ) : (
          <>
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              stroke="currentColor"
              fill="none"
              viewBox="0 0 48 48"
            >
              <path
                d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <div className="mt-4">
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  console.log('🔘 Bouton cliqué, fileInputRef:', fileInputRef.current);
                  if (fileInputRef.current) {
                    fileInputRef.current.click();
                  } else {
                    console.error('❌ fileInputRef.current est null');
                  }
                }}
                className="cursor-pointer inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
              >
                Sélectionner une image
              </button>
              <p className="mt-2 text-sm text-gray-600">
                ou glissez-déposez une image ici
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
