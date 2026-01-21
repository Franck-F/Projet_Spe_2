import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def run_hospital_analysis():
    print("🏥 Starting Deep EDA Part 2: Hospital Analysis & Domain Shift")
    
    root_dir = Path("data/raw/wilds/camelyon17_v1.0")
    patches_dir = root_dir / "patches"
    metadata_path = root_dir / "metadata.csv"
    
    if not metadata_path.exists():
        print("❌ Metadata not found!")
        return

    df = pd.read_csv(metadata_path, index_col=0)
    
    # Rename for clarity if needed (Wilds uses 'center', 'tumor')
    # center = Hospital ID (0-4)
    # tumor = Label (0=Normal, 1=Tumor)
    
    report_lines = ["# Rapport EDA : Analyse par Hôpital & Domain Shift\n"]
    
    # 1. Distribution Quantitative
    report_lines.append("## 1. Distribution Quantitative par Hôpital")
    
    # Patch counts
    patch_counts = df['center'].value_counts().sort_index()
    # Patient counts (patients are unique to centers usually)
    patient_counts = df.groupby('center')['patient'].nunique().sort_index()
    
    report_lines.append("| Hôpital (Center) | Patchs | % Patchs | Patients | Patchs/Patient (Moy) |")
    report_lines.append("|---|---|---|---|---|")
    
    total_patches = len(df)
    
    for center in patch_counts.index:
        n_patches = patch_counts[center]
        n_patients = patient_counts[center]
        pct = (n_patches / total_patches) * 100
        ratio = n_patches / n_patients if n_patients > 0 else 0
        report_lines.append(f"| {center} | {n_patches:,} | {pct:.1f}% | {n_patients} | {ratio:.1f} |")
        
    # 2. Label Imbalance per Hospital
    report_lines.append("\n## 2. Déséquilibre des Labels (Normal vs Tumeur)")
    report_lines.append("| Hôpital | Normal (0) | Tumeur (1) | % Tumeur | Ratio N:T |")
    report_lines.append("|---|---|---|---|---|")
    
    for center in patch_counts.index:
        sub_df = df[df['center'] == center]
        n_normal = len(sub_df[sub_df['tumor'] == 0])
        n_tumor = len(sub_df[sub_df['tumor'] == 1])
        pct_tumor = (n_tumor / len(sub_df)) * 100 if len(sub_df) > 0 else 0
        ratio_nt = f"{n_normal/n_tumor:.1f}:1" if n_tumor > 0 else "Inf"
        
        report_lines.append(f"| {center} | {n_normal:,} | {n_tumor:,} | {pct_tumor:.2f}% | {ratio_nt} |")

    # 3. Domain Shift Analysis (RGB Stats)
    report_lines.append("\n## 3. Analyse du Domain Shift (Statistiques RGB)")
    report_lines.append("> Basé sur un échantillon aléatoire de 500 patchs par hôpital.\n")
    
    report_lines.append("| Hôpital | Mean R | Mean G | Mean B | Luma (Brightness) |")
    report_lines.append("|---|---|---|---|---|")
    
    n_samples = 500
    print(f"Sampling {n_samples} patches per hospital for RGB analysis...")
    
    stats_per_hospital = {}
    
    for center in patch_counts.index:
        # Sample patch indices
        indices = df[df['center'] == center].sample(n_samples, random_state=42).index
        
        r_means, g_means, b_means = [], [], []
        
        for idx in tqdm(indices, desc=f"Hosp {center}", leave=False):
            row = df.loc[idx]
            # Construct filename: patch_patient_XXX_node_Y_x_AAAA_y_BBBB.png
            # patient is int, node is int. coords are int.
            fname = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
            # Subfolder: patient_XXX_node_Y
            subfolder = f"patient_{row['patient']:03d}_node_{row['node']}"
            
            img_path = patches_dir / subfolder / fname
            
            try:
                with Image.open(img_path) as img:
                    img_np = np.array(img.convert('RGB'))
                    r_means.append(np.mean(img_np[:,:,0]))
                    g_means.append(np.mean(img_np[:,:,1]))
                    b_means.append(np.mean(img_np[:,:,2]))
            except Exception as e:
                # print(f"Error reading {img_path}: {e}")
                pass
        
        if r_means:
            avg_r = np.mean(r_means)
            avg_g = np.mean(g_means)
            avg_b = np.mean(b_means)
            luma = 0.299*avg_r + 0.587*avg_g + 0.114*avg_b
            
            report_lines.append(f"| {center} | {avg_r:.1f} | {avg_g:.1f} | {avg_b:.1f} | {luma:.1f} |")
            stats_per_hospital[center] = (avg_r, avg_g, avg_b)
        else:
            report_lines.append(f"| {center} | N/A | N/A | N/A | N/A |")

    # Shift Conclusion
    report_lines.append("\n## 4. Observations sur le Shift\n")
    if stats_per_hospital:
        means = np.array(list(stats_per_hospital.values()))
        # Simple Euclidean distance from global mean to find outlier
        global_mean = np.mean(means, axis=0)
        distances = {k: np.linalg.norm(v - global_mean) for k, v in stats_per_hospital.items()}
        sorted_dist = sorted(distances.items(), key=lambda x: x[1], reverse=True)
        max_shift_center = sorted_dist[0][0]
        
        report_lines.append(f"- **Hôpital le plus divergent** : Center {max_shift_center} (Distance au centre moyen: {distances[max_shift_center]:.2f})")
        report_lines.append("- Les différences de luminosité et de balance des couleurs indiquent la nécessité d'une **Normalisation** ou d'une **Augmentation de couleur** robuste.")

    # Save
    output_path = Path("reports/eda_hospital_analysis.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"✅ Hospital analysis complete. Report saved to {output_path}")

if __name__ == "__main__":
    run_hospital_analysis()
