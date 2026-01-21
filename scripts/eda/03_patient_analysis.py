
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def run_patient_analysis():
    print("👤 Starting Deep EDA Part 3: Patient Analysis")
    
    root_dir = Path("data/raw/wilds/camelyon17_v1.0")
    metadata_path = root_dir / "metadata.csv"
    
    if not metadata_path.exists():
        print("❌ Metadata not found!")
        return

    df = pd.read_csv(metadata_path, index_col=0)
    
    report_lines = ["# Rapport EDA : Analyse Niveau Patient\n"]
    
    # 1. Métadonnées Patients
    report_lines.append("## 1. Statistiques Globales Patients")
    
    num_patients = df['patient'].nunique()
    report_lines.append(f"- **Nombre total de patients uniques**: {num_patients}")
    
    # Group by patient
    patient_stats = df.groupby('patient').agg({
        'tumor': ['count', 'sum', 'mean'],
        'node': 'nunique',
        'center': 'nunique'
    })
    patient_stats.columns = ['n_patches', 'n_tumor', 'tumor_fraction', 'n_nodes', 'n_centers']
    
    # Descriptive stats
    desc = patient_stats['n_patches'].describe()
    report_lines.append(f"- **Patchs par patient**:")
    report_lines.append(f"  - Min: {desc['min']:.0f}")
    report_lines.append(f"  - Max: {desc['max']:.0f}")
    report_lines.append(f"  - Moyenne: {desc['mean']:.1f}")
    report_lines.append(f"  - Médiane: {desc['50%']:.1f}")
    
    # Check if patients belong to multiple centers
    multi_center_patients = patient_stats[patient_stats['n_centers'] > 1]
    if len(multi_center_patients) > 0:
        report_lines.append(f"- ⚠️ **ATTENTION**: {len(multi_center_patients)} patients associés à plusieurs hôpitaux!")
    else:
        report_lines.append(f"- **Intégrité Hôpital**: ✅ Chaque patient est unique à un seul hôpital.")

    # 2. Distribution des Patchs Tumoraux par Patient
    report_lines.append("\n## 2. Charge Tumorale par Patient")
    
    # Patients with 0 tumor patches
    n_healthy_patients = len(patient_stats[patient_stats['n_tumor'] == 0])
    pct_healthy = (n_healthy_patients / num_patients) * 100
    report_lines.append(f"- **Patients sains (0 patchs tumoraux)**: {n_healthy_patients} ({pct_healthy:.1f}%)")
    
    # Patients with tumor patches
    n_sick_patients = num_patients - n_healthy_patients
    pct_sick = (n_sick_patients / num_patients) * 100
    report_lines.append(f"- **Patients avec tumeur**: {n_sick_patients} ({pct_sick:.1f}%)")
    
    # Tumor fraction stats for sick patients
    sick_df = patient_stats[patient_stats['n_tumor'] > 0]
    if not sick_df.empty:
        tf_desc = sick_df['tumor_fraction'].describe()
        report_lines.append(f"- **Fraction tumorale (chez patients malades)**:")
        report_lines.append(f"  - Min: {tf_desc['min']:.4f} ({tf_desc['min']*100:.2f}%)")
        report_lines.append(f"  - Max: {tf_desc['max']:.4f} ({tf_desc['max']*100:.2f}%)")
        report_lines.append(f"  - Moyenne: {tf_desc['mean']:.4f} ({tf_desc['mean']*100:.2f}%)")
    
    # 3. Distribution Nodes
    report_lines.append("\n## 3. Analyse des Nodes (Ganglions)")
    node_desc = patient_stats['n_nodes'].describe()
    report_lines.append(f"- **Nodes par patient**:")
    report_lines.append(f"  - Moyenne: {node_desc['mean']:.1f}")
    report_lines.append(f"  - Max: {node_desc['max']:.0f}")

    # Save Report
    output_path = Path("reports/eda_patient_analysis.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ Patient analysis complete. Report saved to {output_path}")

if __name__ == "__main__":
    run_patient_analysis()
