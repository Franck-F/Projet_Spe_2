import json
from pathlib import Path

notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\03_modeling_patch.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

target_source_part = "train_transforms = transforms.Compose(["

for cell in cells:
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        if any(target_source_part in line for line in source_lines):
            # We found the cell. Let's rewrite the transforms completely.
            new_source = [
                "# Transformations : Aucune (Juste conversion Tensor)\n",
                "train_transforms = transforms.Compose([\n",
                "    transforms.ToTensor()\n",
                "])\n",
                "\n",
                "val_test_transforms = transforms.Compose([\n",
                "    transforms.ToTensor()\n",
                "])\n",
                "\n",
                "# Instanciation\n",
                "train_dataset = CamelyonDataset(df_train, transform=train_transforms)\n",
                "val_dataset = CamelyonDataset(df_val, transform=val_test_transforms)\n",
                "test_dataset = CamelyonDataset(df_test, transform=val_test_transforms)\n",
                "\n",
                "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)\n",
                "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)\n",
                "test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)\n",
                "\n",
                "print(\"DataLoaders prÃªts (Sans transformations).\")"
            ]
            cell['source'] = new_source
            print("Transforms successfully removed.")
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
