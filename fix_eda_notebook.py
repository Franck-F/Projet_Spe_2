import json
import re
from pathlib import Path

notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\EDA_5000.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
# Find the cell that has "spatial_bias_correction_code" id (added in previous step)
target_id = "spatial_bias_correction_code"
target_index = -1

for i, cell in enumerate(cells):
    if cell.get('id') == target_id:
        target_index = i
        break

if target_index != -1:
    # Update the source of this cell to be robust against "Unknown" values
    cells[target_index]['source'] = [
        "# PrÃ©paration : On utilise le DataFrame construit prÃ©cÃ©demment\n",
        "# CRITIQUE : S'assurer que 'tumor' est numÃ©rique et ne contient pas de string 'Unknown' pour les calculs\n",
        "df = df_spatial.copy()\n",
        "df['tumor'] = pd.to_numeric(df['tumor'], errors='coerce')\n",
        "# On supprime les NaN pour l'analyse stats (ce sont les 'Unknown')\n",
        "df = df.dropna(subset=['tumor']).copy()\n",
        "\n",
        "# Fonctions de correction de biais\n",
        "def spatial_uniformity(df):\n",
        "    df = df.copy()\n",
        "    df['grid_x'] = pd.cut(df['x_coord'], bins=10, labels=False)\n",
        "    df['grid_y'] = pd.cut(df['y_coord'], bins=10, labels=False)\n",
        "    # On vÃ©rifie la densitÃ© moyenne de 'tumor' par zone\n",
        "    if 'tumor' not in df.columns: return 0\n",
        "    density = df.groupby(['grid_x', 'grid_y'])['tumor'].mean()\n",
        "    return density.var() # Plus la variance est faible, plus c'est uniforme\n",
        "\n",
        "# Test 1 : Grille 4x4\n",
        "def v1_4x4(df):\n",
        "    df = df.copy()\n",
        "    df['grid_x'] = pd.cut(df['x_coord'], bins=4, labels=False)\n",
        "    df['grid_y'] = pd.cut(df['y_coord'], bins=4, labels=False)\n",
        "    subsample = []\n",
        "    for center in df['center'].unique():\n",
        "        for tumor in [0, 1]:\n",
        "            df_target = df[(df['center'] == center) & (df['tumor'] == tumor)]\n",
        "            for gx in range(4):\n",
        "                for gy in range(4):\n",
        "                    zone = df_target[(df_target['grid_x'] == gx) & (df_target['grid_y'] == gy)]\n",
        "                    if len(zone) > 0: \n",
        "                        subsample.append(zone.sample(1))\n",
        "    if not subsample: return pd.DataFrame(columns=df.columns)\n",
        "    df_v1 = pd.concat(subsample).drop_duplicates()\n",
        "    return df_v1.sample(min(5000, len(df_v1)), random_state=42)\n",
        "\n",
        "# Test 2 : Grille 8x8\n",
        "def v2_8x8(df):\n",
        "    df = df.copy()\n",
        "    df['grid_x'] = pd.cut(df['x_coord'], bins=8, labels=False)\n",
        "    df['grid_y'] = pd.cut(df['y_coord'], bins=8, labels=False)\n",
        "    subsample = []\n",
        "    for center in df['center'].unique():\n",
        "        for tumor in [0, 1]:\n",
        "            df_target = df[(df['center'] == center) & (df['tumor'] == tumor)]\n",
        "            for gx in range(8):\n",
        "                for gy in range(8):\n",
        "                    zone = df_target[(df_target['grid_x'] == gx) & (df_target['grid_y'] == gy)]\n",
        "                    if len(zone) > 0: \n",
        "                        subsample.append(zone.sample(1))\n",
        "    if not subsample: return pd.DataFrame(columns=df.columns)\n",
        "    df_v2 = pd.concat(subsample).drop_duplicates()\n",
        "    return df_v2.sample(min(5000, len(df_v2)), random_state=42)\n",
        "\n",
        "# Test 3 : Grille 10x10\n",
        "def v3_10x10(df):\n",
        "    df = df.copy()\n",
        "    df['grid_x'] = pd.cut(df['x_coord'], bins=10, labels=False)\n",
        "    df['grid_y'] = pd.cut(df['y_coord'], bins=10, labels=False)\n",
        "    subsample = []\n",
        "    for center in df['center'].unique():\n",
        "        for tumor in [0, 1]:\n",
        "            df_target = df[(df['center'] == center) & (df['tumor'] == tumor)]\n",
        "            for gx in range(10):\n",
        "                for gy in range(10):\n",
        "                    zone = df_target[(df_target['grid_x'] == gx) & (df_target['grid_y'] == gy)]\n",
        "                    if len(zone) > 0: \n",
        "                        subsample.append(zone.sample(1))\n",
        "    if not subsample: return pd.DataFrame(columns=df.columns)\n",
        "    df_v3 = pd.concat(subsample).drop_duplicates()\n",
        "    return df_v3.sample(min(5000, len(df_v3)), random_state=42)\n",
        "\n",
        "# ExÃ©cution des tests\n",
        "print(\"GÃ©nÃ©ration des variants de dataset...\")\n",
        "df_v1 = v1_4x4(df)\n",
        "df_v2 = v2_8x8(df)\n",
        "df_v3 = v3_10x10(df)\n",
        "\n",
        "# Comparaison\n",
        "results = pd.DataFrame({\n",
        "    'Version': ['AVANT', 'V1 4x4', 'V2 8x8', 'V3 10x10'],\n",
        "    'N patches': [len(df), len(df_v1), len(df_v2), len(df_v3)],\n",
        "    'Balance %': [df['tumor'].mean(), df_v1['tumor'].mean(), df_v2['tumor'].mean(), df_v3['tumor'].mean()],\n",
        "    'Variance spatiale': [\n",
        "        spatial_uniformity(df), spatial_uniformity(df_v1), \n",
        "        spatial_uniformity(df_v2), spatial_uniformity(df_v3)\n",
        "    ]\n",
        "})\n",
        "\n",
        "print(\"Classement des mÃ©thodes :\")\n",
        "display(results.round(3))\n",
        "\n",
        "best_idx = results['Variance spatiale'].idxmin()\n",
        "best_version = results.iloc[best_idx]\n",
        "print(f\"\\n MEILLEURE : {best_version['Version']} (variance {best_version['Variance spatiale']:.3f})\")"
    ]
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook cell patched successfully.")
else:
    print(f"Target cell with id {target_id} not found.")
