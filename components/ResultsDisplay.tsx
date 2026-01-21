'use client';

interface ResultsDisplayProps {
  results: any;
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  if (!results) {
    return (
      <div className="bg-white rounded-lg shadow-xl p-6">
        <div className="text-center text-gray-500">
          <p>Aucun résultat disponible</p>
        </div>
      </div>
    );
  }
  
  if (results.error) {
    return (
      <div className="bg-red-50 border-2 border-red-200 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-red-900 mb-4">❌ Erreur</h2>
        <p className="text-red-700">{results.error}</p>
      </div>
    );
  }

  // Vérifier si les métadonnées de cancer sont disponibles
  const hasMetadata = results.camelyon17 && results.camelyon17.tumor !== undefined;
  const hasCancer = hasMetadata && results.camelyon17.tumor === 1;
  const tumorLabel = hasMetadata ? results.camelyon17.tumor : 'N/A';
  const metadata = results.camelyon17
    ? [
        results.camelyon17.tumor ?? 'N/A',
        results.camelyon17.slide ?? 'N/A',
        results.camelyon17.center ?? 'N/A',
        results.camelyon17.split ?? 'N/A',
      ]
    : [];

  return (
    <div className="bg-white rounded-lg shadow-xl p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Résultats de l'analyse</h2>

      {/* Résultat principal */}
      <div
        className={`rounded-lg p-6 mb-6 ${
          !hasMetadata
            ? 'bg-yellow-50 border-2 border-yellow-200'
            : hasCancer
            ? 'bg-red-50 border-2 border-red-200'
            : 'bg-green-50 border-2 border-green-200'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h3 className="text-xl font-semibold mb-2">
              {!hasMetadata 
                ? '⚠️ Métadonnées de cancer non disponibles'
                : hasCancer 
                ? '⚠️ CANCER DÉTECTÉ' 
                : '✓ Pas de cancer'}
            </h3>
            <div className="text-sm text-gray-700 space-y-1">
              <p>
                <span className="font-semibold">Label:</span> {tumorLabel} 
                {tumorLabel === 1 ? ' (Cancer ⚠️)' : tumorLabel === 0 ? ' (Pas de cancer ✓)' : ' (Non disponible)'}
              </p>
              {hasMetadata && (
                <>
                  <p>
                    <span className="font-semibold">Metadata:</span> {JSON.stringify(metadata)}
                  </p>
                  {metadata.length === 4 && (
                    <div className="mt-2 text-xs text-gray-600 bg-white/50 p-2 rounded">
                      <div><strong>Tumor:</strong> {metadata[0]} {metadata[0] === 1 ? '(Cancer ⚠️)' : metadata[0] === 0 ? '(Normal ✓)' : '(N/A)'}</div>
                      <div><strong>Slide:</strong> {metadata[1]} (identifiant de la slide histologique)</div>
                      <div><strong>Center:</strong> {metadata[2]} (identifiant du centre médical)</div>
                      <div><strong>Split:</strong> {metadata[3]} {metadata[3] === '0' ? '(Train)' : metadata[3] === '1' ? '(Test)' : metadata[3] === '2' ? '(Validation)' : ''}</div>
                    </div>
                  )}
                </>
              )}
              {!hasMetadata && (
                <div className="mt-2 text-xs text-yellow-700 bg-yellow-100/50 p-2 rounded">
                  <p>ℹ️ L'analyse a été effectuée sans utiliser metadata.csv.</p>
                  <p>Pour détecter le cancer, activez l'option "Utiliser metadata.csv" avant d'uploader l'image.</p>
                </div>
              )}
            </div>
          </div>
          <div
            className={`text-4xl font-bold ml-4 ${
              !hasMetadata
                ? 'text-yellow-600'
                : hasCancer 
                ? 'text-red-600' 
                : 'text-green-600'
            }`}
          >
            {tumorLabel}
          </div>
        </div>
      </div>

      {/* Détails de l'image */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold text-gray-700 mb-2">Informations de l'image</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>Format: {results.image?.format}</li>
            <li>Dimensions: {results.image?.width} × {results.image?.height} pixels</li>
            <li>Mode: {results.image?.mode}</li>
            <li>Taille: {(results.file?.size / 1024).toFixed(2)} KB</li>
          </ul>
        </div>

        {results.camelyon17 && (
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-700 mb-2">Métadonnées Camelyon17</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>Patient: {results.camelyon17.patient}</li>
              <li>Node: {results.camelyon17.node}</li>
              <li>Coordonnées: X={results.camelyon17.x}, Y={results.camelyon17.y}</li>
              <li>Slide: {results.camelyon17.slide}</li>
              <li>Center: {results.camelyon17.center}</li>
              <li>Split: {results.camelyon17.split}</li>
            </ul>
          </div>
        )}
      </div>

      {/* Statistiques des pixels */}
      {results.pixels && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold text-gray-700 mb-3">Statistiques des pixels</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(results.pixels).map(([channel, stats]: [string, any]) => (
              <div key={channel} className="bg-white rounded p-3">
                <h5 className="font-medium text-gray-700 mb-2">
                  {channel === 'channel_0'
                    ? 'Rouge'
                    : channel === 'channel_1'
                    ? 'Vert'
                    : channel === 'channel_2'
                    ? 'Bleu'
                    : channel === 'channel_3'
                    ? 'Alpha'
                    : channel}
                </h5>
                <ul className="text-xs text-gray-600 space-y-1">
                  <li>Min: {stats.min}</li>
                  <li>Max: {stats.max}</li>
                  <li>Moyenne: {stats.mean?.toFixed(2)}</li>
                  <li>Écart-type: {stats.std?.toFixed(2)}</li>
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* JSON complet (collapsible) */}
      <details className="mt-6">
        <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
          Voir les métadonnées complètes (JSON)
        </summary>
        <pre className="mt-2 p-4 bg-gray-900 text-green-400 rounded-lg overflow-auto text-xs">
          {JSON.stringify(results, null, 2)}
        </pre>
      </details>
    </div>
  );
}
