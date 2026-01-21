import pandas as pd
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def run_inventory():
    print("📋 Starting Deep EDA Part 1: Inventory & Structure")
    
    root_dir = Path("data/raw/wilds/camelyon17_v1.0")
    patches_dir = root_dir / "patches"
    metadata_path = root_dir / "metadata.csv"
    
    report_lines = ["# Rapport d'Inventaire EDA - CAMELYON17\n"]
    
    # 1. File Structure
    print("Checking file structure...")
    if not root_dir.exists():
        print(f"❌ Root directory missed: {root_dir}")
        return
    
    report_lines.append("## 1. Structure des Fichiers")
    report_lines.append(f"- **Root**: `{root_dir}`")
    
    if patches_dir.exists():
        # Recursive count of all PNG files
        num_files = sum(1 for _ in patches_dir.rglob("*.png"))
        report_lines.append(f"- **Dossier Patches**: ✅ Présent (Structure hiérarchique détectée)")
        report_lines.append(f"- **Nombre total de patchs (.png)**: {num_files}")
        print(f"Found {num_files} patch files (recursive scan).")
    else:
        report_lines.append("- **Dossier Patches**: ❌ ABSENT")
        print("❌ Patches directory missing!")
        
    csv_files = list(root_dir.glob("*.csv"))
    report_lines.append(f"- **Fichiers CSV trouvés**: {', '.join([f.name for f in csv_files])}")
    
    # 2. Metadata Analysis
    print("Analyzing metadata.csv...")
    if metadata_path.exists():
        df = pd.read_csv(metadata_path, index_col=0)
        report_lines.append("\n## 2. Analyse Metadata")
        report_lines.append(f"- **Dimensions**: {df.shape}")
        report_lines.append(f"- **Colonnes**: {', '.join(df.columns)}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() == 0:
            report_lines.append("- **Valeurs manquantes**: ✅ Aucune")
        else:
            report_lines.append("- **Valeurs manquantes**: ⚠️")
            for col, val in missing.items():
                if val > 0:
                    report_lines.append(f"  - {col}: {val}")
        
        # Consistency check
        if patches_dir.exists():
            # Check if expected number of patches matches metadata
            # Assuming filenames are formatted or indexed. 
            # In WILDS C17, metadata index corresponds to patch? 
            # Or is there a filename column? 
            # Based on previous view, there isn't a filename column, just patient, node, coords...
            # But the dataset loader constructs filenames from these? Or uses index?
            # Usually index in metadata corresponds to file `patch_{index}.png` or similar?
            # Let's assume equality of count is a good first check.
            if len(df) == num_files:
                report_lines.append(f"- **Cohérence**: ✅ {len(df)} entrées métadonnées == {num_files} fichiers physiques")
            else:
                report_lines.append(f"- **Cohérence**: ⚠️ {len(df)} entrées vs {num_files} fichiers")
    else:
        report_lines.append("\n## 2. Analyse Metadata: ❌ Fichier manquant")

    # 3. Splits
    # Usually WILDS has a 'split' column in metadata.csv rather than separate folders
    if 'split' in df.columns:
        print("Analyzing splits...")
        report_lines.append("\n## 3. Analyse des Splits")
        split_counts = df['split'].value_counts().sort_index()
        # 0=train, 1=val (id), 2=test, 3=val (ood)? Need to verify WILDS mapping
        # Ideally we map them 
        # From WILDS doc: 0=train, 1=val_id, 2=val_ood, 3=test .... actually usually:
        # train=0, val=1, test=2. Let's list what we find.
        
        for split_id, count in split_counts.items():
            report_lines.append(f"- **Split {split_id}**: {count} patchs ({count/len(df)*100:.1f}%)")
    
    # Save Report
    output_path = Path("reports/eda_inventory.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ Inventory finished. Report saved to {output_path}")

if __name__ == "__main__":
    run_inventory()
