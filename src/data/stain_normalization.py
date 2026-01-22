"""
Module de Stain Normalization pour harmonisation des couleurs entre hôpitaux.
Implémente les méthodes Macenko et Reinhard.
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def stain_normalization_macenko(img, target_concentrations=None, target_stains=None):
    """
    Méthode Macenko : Estimation des vecteurs de couleur (Hématoxyline et Éosine).
    
    Paramètres:
    -----------
    img : np.ndarray
        Image RGB numpy array (shape: H, W, 3)
    target_concentrations : np.ndarray, optional
        Concentrations cibles pour chaque colorant
    target_stains : np.ndarray, optional
        Vecteurs de colorants cibles (shape: 3, 2)
    
    Retour:
    -------
    img_norm : np.ndarray
        Image normalisée
    target_concentrations : np.ndarray
        Concentrations utilisées pour la normalisation
    target_stains : np.ndarray
        Vecteurs de colorants utilisés
    """
    # Conversion en float et clipping
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.clip(img, 0.001, 1.0)
    
    # Optical density (OD) : -log(I)
    od = -np.log(img)
    od_reshape = od.reshape(-1, 3)
    
    # Estimation des vecteurs de couleur via PCA
    pca = SklearnPCA(n_components=2)
    pca.fit(od_reshape)
    
    stains = pca.components_.T
    stains = stains / (np.linalg.norm(stains, axis=0, keepdims=True) + 1e-8)
    
    # Concentration des colorants
    concentrations = np.dot(od_reshape, stains)
    concentrations = np.clip(concentrations, 0, None)
    
    # Cibles
    if target_concentrations is None:
        target_concentrations = np.percentile(concentrations, 99, axis=0)
    if target_stains is None:
        target_stains = stains
    
    # Normalisation
    concentrations_norm = concentrations / (target_concentrations + 1e-8)
    od_norm = np.dot(concentrations_norm, target_stains.T)
    
    # Conversion de OD vers intensité
    img_norm = np.exp(-od_norm)
    img_norm = np.clip(img_norm * 255, 0, 255).astype(np.uint8)
    
    return img_norm.reshape(img.shape), target_concentrations, target_stains


def stain_normalization_reinhard(img, reference_mean=None, reference_std=None):
    """
    Méthode Reinhard : Normalisation basée sur moyenne/variance RGB.
    Plus simple et rapide que Macenko.
    
    Paramètres:
    -----------
    img : np.ndarray
        Image RGB numpy array (shape: H, W, 3)
    reference_mean : np.ndarray, optional
        Moyenne RGB cible (shape: 3,)
    reference_std : np.ndarray, optional
        Écart-type RGB cible (shape: 3,)
    
    Retour:
    -------
    img_norm : np.ndarray
        Image normalisée
    reference_mean : np.ndarray
        Moyenne utilisée
    reference_std : np.ndarray
        Écart-type utilisé
    """
    img = np.array(img, dtype=np.float32)
    
    # Cibles par défaut
    if reference_mean is None:
        reference_mean = np.array([150, 130, 145])
    if reference_std is None:
        reference_std = np.array([40, 40, 35])
    
    # Stats actuelles
    current_mean = np.mean(img, axis=(0, 1))
    current_std = np.std(img, axis=(0, 1))
    
    # Normalisation
    img_norm = np.zeros_like(img)
    for i in range(3):
        if current_std[i] > 0:
            img_norm[:, :, i] = (img[:, :, i] - current_mean[i]) * (reference_std[i] / current_std[i]) + reference_mean[i]
        else:
            img_norm[:, :, i] = img[:, :, i]
    
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    return img_norm, reference_mean, reference_std


def compute_template_reference(df, patches_dir, n_samples=500, method='reinhard'):
    """
    Calcule un template de référence global à partir d'un échantillon de patchs.
    
    Paramètres:
    -----------
    df : pd.DataFrame
        DataFrame avec métadonnées des patchs
    patches_dir : Path
        Répertoire contenant les patchs
    n_samples : int
        Nombre d'images à analyser (par défaut 500)
    method : str
        'reinhard' ou 'macenko'
    
    Retour:
    -------
    template_mean : np.ndarray
        Moyenne RGB globale (si Reinhard)
    template_std : np.ndarray
        Écart-type RGB global (si Reinhard)
    """
    sample = df.sample(min(n_samples, len(df)), random_state=42)
    
    means_list = []
    stds_list = []
    
    print(f"Calcul du template ({method})...")
    
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        fname = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        path = patches_dir / f"patient_{row['patient']:03d}_node_{row['node']}" / fname
        
        if path.exists():
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32)
            
            means_list.append(np.mean(img, axis=(0, 1)))
            stds_list.append(np.std(img, axis=(0, 1)))
    
    if not means_list:
        print("⚠️  Aucune image trouvée!")
        return None, None
    
    means_array = np.array(means_list)
    stds_array = np.array(stds_list)
    
    # Utiliser la médiane (plus robuste)
    template_mean = np.median(means_array, axis=0)
    template_std = np.median(stds_array, axis=0)
    
    print(f"✓ Template calculé:")
    print(f"  Moyenne RGB: {template_mean}")
    print(f"  Écart-type RGB: {template_std}")
    
    return template_mean, template_std


def apply_stain_normalization_to_dataset(df, patches_dir, output_dir, 
                                          template_mean, template_std, 
                                          method='reinhard', batch_size=100):
    """
    Applique la normalisation de couleur à tout le dataset et sauvegarde les images.
    
    Paramètres:
    -----------
    df : pd.DataFrame
        DataFrame avec métadonnées
    patches_dir : Path
        Répertoire source
    output_dir : Path
        Répertoire de sortie
    template_mean : np.ndarray
        Moyenne cible
    template_std : np.ndarray
        Écart-type cible
    method : str
        'reinhard' ou 'macenko'
    batch_size : int
        Nombre d'images à traiter avant affichage du progress
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nNormalisation du dataset ({method})...")
    processed = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        src_path = patches_dir / f"patient_{row['patient']:03d}_node_{row['node']}" / fname
        
        if not src_path.exists():
            failed += 1
            continue
        
        try:
            img = cv2.imread(str(src_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if method == 'reinhard':
                img_norm, _, _ = stain_normalization_reinhard(img, template_mean, template_std)
            else:  # macenko
                img_norm, _, _ = stain_normalization_macenko(img)
            
            # Sauvegarder
            output_subdir = output_dir / f"patient_{row['patient']:03d}_node_{row['node']}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_subdir / fname
            img_norm_bgr = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_norm_bgr)
            
            processed += 1
            
        except Exception as e:
            failed += 1
            print(f"Erreur pour {fname}: {e}")
    
    print(f"\n✓ Normalisation complétée:")
    print(f"  Images traitées: {processed:,}")
    print(f"  Erreurs: {failed:,}")
    print(f"  Sauvegardées dans: {output_dir}")
    
    return processed, failed


def verify_patch_sizes(df, patches_dir, n_check=100):
    """
    Vérifie que tous les patchs ont la même taille.
    
    Paramètres:
    -----------
    df : pd.DataFrame
        DataFrame avec métadonnées
    patches_dir : Path
        Répertoire des patchs
    n_check : int
        Nombre de patchs à vérifier
    """
    patch_sizes = []
    
    for i in range(min(n_check, len(df))):
        row = df.iloc[i]
        fname = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        path = patches_dir / f"patient_{row['patient']:03d}_node_{row['node']}" / fname
        
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                patch_sizes.append(img.shape[:2])  # (hauteur, largeur)
    
    unique_sizes = set(patch_sizes)
    
    print("=" * 60)
    print("VÉRIFICATION DES DIMENSIONS DES PATCHS")
    print("=" * 60)
    print(f"Nombre de patchs vérifiés: {len(patch_sizes)}")
    print(f"Tailles uniques trouvées: {len(unique_sizes)}")
    
    if len(unique_sizes) == 1:
        size = list(unique_sizes)[0]
        print(f"✓ TOUS LES PATCHS ONT LA MÊME TAILLE: {size[0]} × {size[1]} pixels")
    else:
        print(f"✗ ATTENTION: Tailles variables détectées:")
        for size in unique_sizes:
            count = patch_sizes.count(size)
            print(f"   - {size[0]} × {size[1]}: {count} patchs")
    
    return len(unique_sizes) == 1


def remove_empty_patches(df, patches_dir, 
                         intensity_threshold=210, 
                         texture_threshold=15):
    """
    Filtre les patchs avec trop de fond blanc (vides).
    
    Paramètres:
    -----------
    df : pd.DataFrame
        DataFrame avec métadonnées
    patches_dir : Path
        Répertoire des patchs
    intensity_threshold : int
        Seuil d'intensité moyenne (>210 = quasi blanc)
    texture_threshold : int
        Seuil d'écart-type (<15 = pas de texture)
    
    Retour:
    -------
    df_clean : pd.DataFrame
        DataFrame nettoyé (sans patchs vides)
    """
    print(f"\nFiltrage des patchs vides...")
    print(f"Nombre de patchs AVANT: {len(df):,}")
    
    empty_patches = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        path = patches_dir / f"patient_{row['patient']:03d}_node_{row['node']}" / fname
        
        is_empty = False
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                intensity = np.mean(img_gray)
                texture = np.std(img_gray)
                
                is_empty = (intensity > intensity_threshold) and (texture < texture_threshold)
        
        empty_patches.append(is_empty)
    
    df['is_empty'] = empty_patches
    df_clean = df[~df['is_empty']].copy()
    
    n_removed = len(df) - len(df_clean)
    pct_removed = (n_removed / len(df)) * 100
    
    print(f"Patchs vides détectés: {n_removed:,} ({pct_removed:.2f}%)")
    print(f"Nombre de patchs APRÈS: {len(df_clean):,}")
    
    # Impact par hôpital
    print("\nImpact par hôpital:")
    for center in sorted(df['center'].unique()):
        total = len(df[df['center'] == center])
        clean = len(df_clean[df_clean['center'] == center])
        removed = total - clean
        pct = (removed / total) * 100 if total > 0 else 0
        print(f"  Hôpital {center}: {total:,} → {clean:,} (supprimés: {removed}, {pct:.1f}%)")
    
    return df_clean.reset_index(drop=True)
