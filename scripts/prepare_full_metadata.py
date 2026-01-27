import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    root_dir = Path(r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2')
    metadata_path = root_dir / 'data/raw/wilds/camelyon17_v1.0/metadata.csv'
    output_path = root_dir / 'data/processed/df_full_normalized.csv'
    normalized_dir = Path('../data/processed/patches_224x224_normalized') # relative to notebooks/

    print(f"Loading original metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path, index_col=0)
    print(f"Loaded {len(df)} entries.")

    print("Generating normalized paths...")
    # Path format: patient_XXX_node_X/patch_patient_XXX_node_X_x_XXXX_y_YYYY.png
    def get_norm_path(row):
        patient_folder = f"patient_{row['patient']:03d}_node_{int(row['node'])}"
        filename = f"patch_patient_{row['patient']:03d}_node_{int(row['node'])}_x_{int(row['x_coord'])}_y_{int(row['y_coord'])}.png"
        return str(normalized_dir / patient_folder / filename)

    tqdm.pandas(desc="Processing paths")
    df['path_normalized'] = df.progress_apply(get_norm_path, axis=1)

    print(f"Saving new metadata to {output_path}...")
    df.to_csv(output_path, index=True)
    print("Done.")

if __name__ == "__main__":
    main()
