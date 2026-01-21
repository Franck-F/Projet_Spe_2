
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def compute_metrics(args):
    img_path, idx = args
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Dimensions
        h, w, c = img.shape
        
        # Convert to grayscale for some metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Background check (Mean intensity)
        mean_intensity = np.mean(gray)
        
        # 2. Contrast (Std Dev)
        std_intensity = np.std(gray)
        
        # 3. Blurriness (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            'index': idx,
            'height': h,
            'width': w,
            'channels': c,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'blur_var': laplacian_var
        }
    except Exception as e:
        return None

def run_patch_quality_analysis():
    print("🔬 Starting Deep EDA Part 4: Patch Quality Analysis")
    
    root_dir = Path("data/raw/wilds/camelyon17_v1.0")
    patches_dir = root_dir / "patches"
    metadata_path = root_dir / "metadata.csv"
    
    if not metadata_path.exists():
        print("❌ Metadata not found!")
        return

    df = pd.read_csv(metadata_path, index_col=0)
    
    # Sample patches (e.g., 5000 for speed)
    n_sample = 5000
    print(f"Sampling {n_sample} patches for quality check...")
    sample_df = df.sample(n=n_sample, random_state=42)
    
    tasks = []
    for idx, row in sample_df.iterrows():
        fname = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        subfolder = f"patient_{row['patient']:03d}_node_{row['node']}"
        img_path = patches_dir / subfolder / fname
        tasks.append((img_path, idx))
    
    # Run analysis in parallel
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in tqdm(executor.map(compute_metrics, tasks), total=len(tasks)):
            if res:
                results.append(res)
    
    results_df = pd.DataFrame(results)
    
    report_lines = ["# Rapport EDA : Qualité des Patchs\n"]
    report_lines.append(f"> Analyse basée sur un échantillon de {len(results_df)} patchs.\n")
    
    # 1. Dimensions
    report_lines.append("## 1. Dimensions et Format")
    dims = results_df.groupby(['height', 'width', 'channels']).size()
    for d, count in dims.items():
        report_lines.append(f"- **{d}**: {count} patchs ({count/len(results_df)*100:.1f}%)")
    
    if len(dims) > 1:
        report_lines.append("- ⚠️ **ATTENTION**: Dimensions hétérogènes détectées!")
    else:
        report_lines.append("- ✅ Dimensions constantes.")

    # 2. Background / Empty Patches
    # High mean intensity + Low std dev usually means white background
    report_lines.append("\n## 2. Détection de Fond Blanc (Background)")
    
    # Thresholds (tunable)
    white_thresh = 210 # > 210 mean intensity (0-255)
    std_thresh = 15    # < 15 std dev (flat)
    
    empty_candidates = results_df[
        (results_df['mean_intensity'] > white_thresh) & 
        (results_df['std_intensity'] < std_thresh)
    ]
    
    report_lines.append(f"- **Candidats 'Fond Blanc/Vide'** (Mean > {white_thresh}, Std < {std_thresh}):")
    report_lines.append(f"  - Nombre: {len(empty_candidates)}")
    report_lines.append(f"  - Pourcentage: {len(empty_candidates)/len(results_df)*100:.2f}%")
    
    # 3. Blurriness
    report_lines.append("\n## 3. Analyse du Flou (Blur)")
    blur_thresh = 100 # Low variance = blurry (heuristic)
    blurry_candidates = results_df[results_df['blur_var'] < blur_thresh]
    
    report_lines.append(f"- **Candidats 'Flous'** (Laplacian Var < {blur_thresh}):")
    report_lines.append(f"  - Nombre: {len(blurry_candidates)}")
    report_lines.append(f"  - Pourcentage: {len(blurry_candidates)/len(results_df)*100:.2f}%")
    report_lines.append(f"- **Statistiques Blur Variance**:")
    report_lines.append(f"  - Moyenne: {results_df['blur_var'].mean():.1f}")
    report_lines.append(f"  - Médiane: {results_df['blur_var'].median():.1f}")

    # Generate Histogram Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(results_df['mean_intensity'], ax=axes[0], bins=30, kde=True, color='blue')
    axes[0].set_title('Distribution Intensité Moyenne')
    
    sns.histplot(results_df['std_intensity'], ax=axes[1], bins=30, kde=True, color='green')
    axes[1].set_title('Distribution Contraste (Std Dev)')
    
    sns.histplot(results_df['blur_var'], ax=axes[2], bins=30, kde=True, color='red')
    axes[2].set_title('Distribution Netteté (Laplacian Var)')
    axes[2].set_xscale('log') # Log scale mainly for blur var as it varies widely
    
    plot_path = Path("reports/figures/patch_quality_dist.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    report_lines.append(f"\n![Distributions]({plot_path.as_posix()})")
    
    # Save Report
    output_path = Path("reports/eda_patch_quality.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ Patch Quality analysis complete. Report saved to {output_path}")

if __name__ == "__main__":
    run_patch_quality_analysis()
