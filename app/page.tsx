'use client';

import { useState } from 'react';
import ImageUpload from '@/components/ImageUpload';
import ResultsDisplay from '@/components/ResultsDisplay';

export default function Home() {
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalysisComplete = (data: any) => {
    console.log('📊 Résultats reçus dans la page:', data);
    setResults(data);
    setLoading(false);
    
    if (!data) {
      console.warn('⚠️ Aucune donnée reçue');
    }
  };

  const handleUploadStart = () => {
    setLoading(true);
    setResults(null);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🔬 Cancer Image Classifier
          </h1>
          <p className="text-lg text-gray-600">
            Uploadez une image de patch histologique pour détecter la présence de cancer
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-xl p-6 mb-6">
          <ImageUpload 
            onUploadStart={handleUploadStart}
            onAnalysisComplete={handleAnalysisComplete}
          />
        </div>

        {loading && (
          <div className="bg-white rounded-lg shadow-xl p-6">
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
              <span className="ml-4 text-gray-600">Analyse en cours...</span>
            </div>
          </div>
        )}

        {results && (
          <ResultsDisplay results={results} />
        )}
      </div>
    </main>
  );
}
