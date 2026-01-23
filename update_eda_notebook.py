import json
import re
from pathlib import Path

notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\EDA_5000.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
target_id = "2ff22963" # "Analyse biais spaciale"
cutoff_index = -1

for i, cell in enumerate(cells):
    if cell.get('id') == target_id:
        cutoff_index = i
        break

if cutoff_index != -1:
    # Keep cells up to the target (inclusive)
    new_cells = cells[:cutoff_index+1]
    
    # --- PREVIOUS VISUALIZATION CELLS (Recap) ---
    
    # Cell 1: Imports and Load Data
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "spatial_analysis_load_v3",
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import plotly.express as px\n",
            "import plotly.graph_objects as go\n",
            "from pathlib import Path\n",
            "import re\n",
            "\n",
            "# Chemin vers les patchs 224x224\n",
            "PATCHES_DIR = Path('../data/processed/patches_224x224')\n",
            "\n",
            "# 1. Extraction des coordonnÃ©es depuis les fichiers\n",
            "data = []\n",
            "pattern = re.compile(r'patch_patient_(\\d+)_node_(\\d+)_x_(\\d+)_y_(\\d+)\\.png')\n",
            "\n",
            "print(\"Recensement des patchs existants (224x224)...\")\n",
            "image_files = list(PATCHES_DIR.rglob('*.png'))\n",
            "print(f\"Fichiers trouvÃ©s : {len(image_files)}\")\n",
            "\n",
            "for file_path in image_files:\n",
            "    match = pattern.match(file_path.name)\n",
            "    if match:\n",
            "        patient_id, node_id, x, y = match.groups()\n",
            "        data.append({\n",
            "            'patient': int(patient_id),\n",
            "            'node': int(node_id),\n",
            "            'x_coord': int(x),\n",
            "            'y_coord': int(y),\n",
            "            'path': str(file_path)\n",
            "        })\n",
            "\n",
            "df_spatial = pd.DataFrame(data)\n",
            "print(f\"DataFrame spatial construit : {len(df_spatial)} lignes\")"
        ]
    })
    
    # Cell 2: Merge
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "spatial_analysis_merge_v3",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Fusion avec les mÃ©tadonnÃ©es originales pour rÃ©cupÃ©rer le statut 'tumor'\n",
            "if 'df' in locals():\n",
            "    print(\"Fusion avec le DataFrame principal 'df'...\")\n",
            "    for col in ['patient', 'node', 'x_coord', 'y_coord']:\n",
            "        df_spatial[col] = df_spatial[col].astype(int)\n",
            "        if col in df.columns:\n",
            "            df[col] = df[col].astype(int)\n",
            "    \n",
            "    cols_to_merge = ['patient', 'node', 'x_coord', 'y_coord', 'center', 'tumor']\n",
            "    cols_to_merge = [c for c in cols_to_merge if c in df.columns]\n",
            "    \n",
            "    df_spatial = pd.merge(df_spatial, df[cols_to_merge], \n",
            "                          on=['patient', 'node', 'x_coord', 'y_coord'], \n",
            "                          how='left')\n",
            "    \n",
            "    if 'tumor' in df_spatial.columns:\n",
            "        na_tumor = df_spatial['tumor'].isna().sum()\n",
            "        print(f\"Valeurs 'tumor' manquantes : {na_tumor}\")\n",
            "        if na_tumor > 0:\n",
            "            df_spatial['tumor'] = df_spatial['tumor'].fillna('Unknown')\n",
            "    else:\n",
            "        print(\"ATTENTION : La colonne 'tumor' n'a pas pu Ãªtre rÃ©cupÃ©rÃ©e.\")\n",
            "else:\n",
            "    print(\"ERREUR : Le DataFrame 'df' n'est pas dÃ©fini.\")\n"
        ]
    })

    # Cell 3: Viz
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "spatial_analysis_viz_plotly",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Visualisation interactive avec Plotly\n",
            "if 'tumor' in df_spatial.columns and not df_spatial.empty:\n",
            "    def get_tumor_label(val):\n",
            "        if val == 1: return 'Tumeur (Tumor)'\n",
            "        if val == 0: return 'Sain (Normal)'\n",
            "        return str(val)\n",
            "        \n",
            "    df_spatial['Status'] = df_spatial['tumor'].apply(get_tumor_label)\n",
            "\n",
            "    fig = px.scatter(\n",
            "        df_spatial, \n",
            "        x='x_coord', \n",
            "        y='y_coord', \n",
            "        color='Status',\n",
            "        title='<b>Distribution Spatiale des Patchs (224x224)</b><br><i>Analyse de la rÃ©partition Tumeur vs Tissu Sain</i>',\n",
            "        labels={'x_coord': 'X', 'y_coord': 'Y', 'Status': 'Pathologie'},\n",
            "        color_discrete_map={'Sain (Normal)': '#1f77b4', 'Tumeur (Tumor)': '#d62728', 'Unknown': 'gray'},\n",
            "        opacity=0.6,\n",
            "    )\n",
            "    fig.update_yaxes(autorange=\"reversed\")\n",
            "    fig.update_layout(template=\"plotly_white\", height=700)\n",
            "    fig.show()\n"
        ]
    })

    # --- NEW CELLS FOR SPATIAL BIAS CORRECTION ---

    # Markdown Header
    new_cells.append({
        "cell_type": "markdown",
        "id": "spatial_bias_title",
        "metadata": {},
        "source": [
            "### BIAIS SPATIAL : DÃ©tection et Correction\n",
            "\n",
            "**Observation** : Les patches tumoraux peuvent former des amas localisÃ©s (biais de position). \n",
            "Si les tumeurs sont toujours au mÃªme endroit sur les lames, le CNN risque d'apprendre \"position gÃ©ographique = tumeur\" au lieu d'apprendre les caractÃ©ristiques histologiques.\n",
            "\n",
            "Nous allons tester 3 stratÃ©gies de sous-Ã©chantillonnage spatial (Grilles 4x4, 8x8, 10x10) pour uniformiser la distribution."
        ]
    })

    # Code: Definitions and Tests
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "spatial_bias_correction_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# PrÃ©paration : On utilise le DataFrame construit prÃ©cÃ©demment\n",
            "df = df_spatial.copy()\n",
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
            "            for gy in range(4):\n",
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
            "            for gy in range(8):\n",
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
            "            for gy in range(10):\n",
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
    })
    
    # Cell 4: Visualization and Save
    new_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "spatial_bias_viz_save",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualisation des rÃ©sultats post-correction\n",
            "datasets = {'AVANT': df, 'V1 4x4': df_v1, 'V2 8x8': df_v2, 'V3 10x10': df_v3}\n",
            "df_all = pd.concat([d.assign(version=name) for name, d in datasets.items()])\n",
            "\n",
            "fig = px.density_heatmap(\n",
            "    df_all, x='x_coord', y='y_coord', z='tumor',\n",
            "    facet_col='version', facet_col_wrap=2,\n",
            "    histfunc='avg', nbinsx=25, nbinsy=25,\n",
            "    color_continuous_scale='RdYlGn',\n",
            "    title='CORRECTION BIAIS SPATIAL : Comparaison Avant/AprÃ¨s',\n",
            "    category_orders={'version': list(datasets.keys())},\n",
            "    height=800, width=1000\n",
            ")\n",
            "\n",
            "fig.update_layout(coloraxis_colorbar=dict(title='Tumor % (avg)'))\n",
            "fig.show()\n",
            "\n",
            "# Sauvegarde de la meilleure version\n",
            "if best_version['Version'] == 'V2 8x8': df_final = df_v2\n",
            "elif best_version['Version'] == 'V3 10x10': df_final = df_v3\n",
            "else: df_final = df_v1 # Fallback sur V1 si V1 est meilleure ou autre\n",
            "\n",
            "output_path = Path('../data/processed/df_final_5000.csv')\n",
            "output_path.parent.mkdir(parents=True, exist_ok=True)\n",
            "df_final.to_csv(output_path, index=False)\n",
            "print(f\"Meilleure version sauvÃ©e dans {output_path} : {len(df_final)} patches\")"
        ]
    })

    nb['cells'] = new_cells
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully with spatial bias correction.")
else:
    print(f"Target cell with id {target_id} not found.")
