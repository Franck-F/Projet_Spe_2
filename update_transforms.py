import json
from pathlib import Path

notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\03_modeling_patch.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Find the cell containing transforms definition
target_source_part = "train_transforms = transforms.Compose(["

for cell in cells:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        if any(target_source_part in line for line in source_lines):
            # Modify the content to comment out Resize
            new_source = []
            for line in source_lines:
                if "transforms.Resize((96, 96))" in line:
                    # Comment out the line
                    new_source.append(line.replace("transforms.Resize((96, 96))", "# transforms.Resize((96, 96)) # Utilisation de 224x224"))
                else:
                    new_source.append(line)
            cell['source'] = new_source
            print("Transforms cell updated.")
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated to use 224x224 resolution.")
