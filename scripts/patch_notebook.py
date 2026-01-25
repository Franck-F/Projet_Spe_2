import json
import os

notebook_path = r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\modelisation_SimpleCNN_patchs_5000.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Add Visualization cells after the loading cell
# Find the cell that loads the CSV (cell with pd.read_csv(DATA_PATH))
load_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'pd.read_csv(DATA_PATH)' in "".join(cell['source']):
        load_cell_idx = i
        break

if load_cell_idx != -1:
    viz_markdown = {
        "cell_type": "markdown",
        "id": "visualisation_title",
        "metadata": {},
        "source": [
            "# Visualisation d'Échantillons de Patchs\n",
            "*Vérification visuelle du chargement des images*"
        ]
    }
    viz_code = {
        "cell_type": "code",
        "execution_count": None,
        "id": "visualisation_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "\n",
            "def visualize_patches(dataframe, patches_dir, n=5):\n",
            "    fig, axes = plt.subplots(1, n, figsize=(20, 4))\n",
            "    # Sélectionner n patchs aléatoires avec au moins une tumeur si possible\n",
            "    if len(dataframe[dataframe['tumor'] == 1]) > 0:\n",
            "        n_tumor = min(n // 2 + 1, len(dataframe[dataframe['tumor'] == 1]))\n",
            "        sample_tumor = dataframe[dataframe['tumor'] == 1].sample(n_tumor)\n",
            "        sample_normal = dataframe[dataframe['tumor'] == 0].sample(n - n_tumor)\n",
            "        sample_df = pd.concat([sample_tumor, sample_normal]).sample(frac=1)\n",
            "    else:\n",
            "        sample_df = dataframe.sample(n)\n",
            "    \n",
            "    for i, (_, row) in enumerate(sample_df.iterrows()):\n",
            "        # Utilisation de path_normalized si présent, sinon fallback sur reconstruction\n",
            "        img_path = Path(row['path_normalized']) if 'path_normalized' in row else Path(row['path'])\n",
            "        \n",
            "        if not img_path.exists():\n",
            "            # Essayer de reconstruire le chemin\n",
            "            patient_folder = f\"patient_{int(row['patient']):03d}_node_{int(row['node'])}\"\n",
            "            filename = img_path.name\n",
            "            img_path = Path(patches_dir) / patient_folder / filename\n",
            "            \n",
            "        try:\n",
            "            img = Image.open(img_path)\n",
            "            axes[i].imshow(img)\n",
            "            status = \"Tumor\" if row['tumor'] == 1 else \"Normal\"\n",
            "            axes[i].set_title(f\"{status}\\nPatient: {row['patient']}\")\n",
            "            axes[i].axis('off')\n",
            "        except Exception as e:\n",
            "            axes[i].text(0.5, 0.5, f\"Error\\n{img_path.name}\", ha='center')\n",
            "            axes[i].axis('off')\n",
            "            \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "visualize_patches(df, PATCHES_DIR)"
        ]
    }
    nb['cells'].insert(load_cell_idx + 1, viz_markdown)
    nb['cells'].insert(load_cell_idx + 2, viz_code)

# 2. Update the Splitting logic
# Find the cell with "available_train = [c for c in TRAIN_VAL_CENTERS"
split_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'available_train = [c for c in TRAIN_VAL_CENTERS' in "".join(cell['source']):
        split_cell_idx = i
        break

if split_cell_idx != -1:
    nb['cells'][split_cell_idx]['source'] = [
        "# Définir les centres pour Train/Val et Test\n",
        "TRAIN_VAL_CENTERS = [0, 1, 2]  \n",
        "TEST_CENTERS = [3, 4]         \n",
        "\n",
        "# Vérification\n",
        "available_train = [c for c in TRAIN_VAL_CENTERS if c in centers]\n",
        "available_test = [c for c in TEST_CENTERS if c in centers]\n",
        "\n",
        "if not available_test:\n",
        "    print(\"WARNING: Aucun centre de test disponible. Utilisation de split aléatoire.\")\n",
        "    # Fallback sur split aléatoire si moins de 5 centres\n",
        "    from sklearn.model_selection import StratifiedGroupKFold\n",
        "    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)\n",
        "    train_idx, test_idx = next(sgkf.split(df, y=df['tumor'], groups=df['patient']))\n",
        "    df_train_val = df.iloc[train_idx].copy()\n",
        "    df_test = df.iloc[test_idx].copy()\n",
        "else:\n",
        "    # Split basé sur les centres\n",
        "    df_train_val = df[df['center'].isin(available_train)].copy()\n",
        "    df_test = df[df['center'].isin(available_test)].copy()\n",
        "\n",
        "# Split Train/Val (Stratifié par Tumeur ET Groupé par Patient)\n",
        "from sklearn.model_selection import StratifiedGroupKFold\n",
        "sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42) # ~15% val\n",
        "train_idx, val_idx = next(sgkf_val.split(\n",
        "    df_train_val, \n",
        "    y=df_train_val['tumor'], \n",
        "    groups=df_train_val['patient']\n",
        "))\n",
        "df_train = df_train_val.iloc[train_idx].copy()\n",
        "df_val = df_train_val.iloc[val_idx].copy()\n",
        "\n",
        "print(f\"\\nSplit Réalisé :\")\n",
        "print(f\"Train : {len(df_train):,} patchs ({df_train['patient'].nunique()} patients, centres {df_train['center'].unique()})\")\n",
        "print(f\"Val   : {len(df_val):,} patchs ({df_val['patient'].nunique()} patients, centres {df_val['center'].unique()})\")\n",
        "print(f\"Test  : {len(df_test):,} patchs ({df_test['patient'].nunique()} patients, centres {df_test['center'].unique()})\")\n",
        "\n",
        "print(\"\\nDistribution de la variable cible (tumor) :\")\n",
        "dist_data = {\n",
        "    'Train': df_train['tumor'].value_counts(normalize=True).sort_index(),\n",
        "    'Val': df_val['tumor'].value_counts(normalize=True).sort_index(),\n",
        "    'Test': df_test['tumor'].value_counts(normalize=True).sort_index()\n",
        "}\n",
        "dist_df = pd.DataFrame(dist_data)\n",
        "display(dist_df.style.format(\"{:.2%}\"))\n"
    ]

# 3. Ensure DATA_PATH is correct in the config cell
config_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'DATA_PATH = Path' in "".join(cell['source']):
        config_cell_idx = i
        break

if config_cell_idx != -1:
    source = nb['cells'][config_cell_idx]['source']
    for j, line in enumerate(source):
        if 'DATA_PATH = Path' in line:
            source[j] = 'DATA_PATH = Path(\'../data/processed/df_20000.csv\')\\n'
    nb['cells'][config_cell_idx]['source'] = source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Modification terminée.")
