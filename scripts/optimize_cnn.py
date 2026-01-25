import json
import os
from pathlib import Path

notebook_path = r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\modelisation_SimpleCNN_patchs_5000.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Config cell (DATA_PATH and Hyperparameters)
config_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'DATA_PATH = Path' in "".join(cell['source']):
        config_cell_idx = i
        break

if config_cell_idx != -1:
    nb['cells'][config_cell_idx]['source'] = [
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "from tqdm.auto import tqdm\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler\n",
        "import torchvision.transforms as transforms\n",
        "from torchvision import models\n",
        "from PIL import Image\n",
        "from pathlib import Path\n",
        "from sklearn.metrics import (\n",
        "    classification_report, roc_auc_score, confusion_matrix,\n",
        "    precision_recall_curve, f1_score\n",
        ")\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import plotly.graph_objects as go\n",
        "from plotly.subplots import make_subplots\n",
        "import plotly.express as px\n",
        "\n",
        "# Configuration Haute Performance\n",
        "DATA_PATH = Path('../data/processed/df_20000.csv')\n",
        "PATCHES_DIR = Path('../data/processed/patches_224x224_normalized_sample')\n",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "BATCH_SIZE = 64\n",
        "EPOCHS = 15\n",
        "LR = 1e-4\n",
        "WEIGHT_DECAY = 1e-2\n",
        "\n",
        "print(f\"Device: {DEVICE}\")\n",
        "print(f\"Patches directory: {PATCHES_DIR}\")\n"
    ]

# 2. Find and update the Splitting cell
split_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'available_train = [c for c in TRAIN_VAL_CENTERS' in "".join(cell['source']):
        split_cell_idx = i
        break

if split_cell_idx != -1:
    nb['cells'][split_cell_idx]['source'] = [
        "from sklearn.model_selection import StratifiedGroupKFold\n",
        "\n",
        "# Split par Centre (Domain Shift Evaluation)\n",
        "centers = df['center'].unique()\n",
        "TRAIN_VAL_CENTERS = [0, 1, 2]\n",
        "TEST_CENTERS = [3, 4]\n",
        "\n",
        "df_train_val = df[df['center'].isin(TRAIN_VAL_CENTERS)].copy()\n",
        "df_test = df[df['center'].isin(TEST_CENTERS)].copy()\n",
        "\n",
        "# Stratified Group Split pour la validation (~15%)\n",
        "sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)\n",
        "train_idx, val_idx = next(sgkf.split(df_train_val, y=df_train_val['tumor'], groups=df_train_val['patient']))\n",
        "\n",
        "df_train = df_train_val.iloc[train_idx].copy()\n",
        "df_val = df_train_val.iloc[val_idx].copy()\n",
        "\n",
        "print(f\"\\nSplit Réalisé (Stratified by Tumor, Grouped by Patient) :\")\n",
        "print(f\"Train : {len(df_train):,} patchs ({df_train['patient'].nunique()} patients)\")\n",
        "print(f\"Val   : {len(df_val):,} patchs ({df_val['patient'].nunique()} patients)\")\n",
        "print(f\"Test  : {len(df_test):,} patchs ({df_test['patient'].nunique()} patients)\")\n",
        "\n",
        "print(\"\\nDistribution de la variable cible (tumor) :\")\n",
        "dist_df = pd.DataFrame({\n",
        "    'Train': df_train['tumor'].value_counts(normalize=True).sort_index(),\n",
        "    'Val': df_val['tumor'].value_counts(normalize=True).sort_index(),\n",
        "    'Test': df_test['tumor'].value_counts(normalize=True).sort_index()\n",
        "})\n",
        "display(dist_df.style.format(\"{:.2%}\"))\n"
    ]

# 3. Update SimpleCNN definition
cnn_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'class SimpleCNN' in "".join(cell['source']):
        cnn_cell_idx = i
        break

if cnn_cell_idx != -1:
    nb['cells'][cnn_cell_idx]['source'] = [
        "class SimpleCNN(nn.Module):\n",
        "    \"\"\"CNN Haute Performance basé sur un backbone pré-entraîné (Transfer Learning)\"\"\"\n",
        "    \n",
        "    def __init__(self, num_classes=1):\n",
        "        super(SimpleCNN, self).__init__()\n",
        "        \n",
        "        # Utilisation de ResNet-18 comme extracteur de caractéristiques (CNN pré-entraîné)\n",
        "        # Ce moteur permet d'atteindre de hautes performances rapidement\n",
        "        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)\n",
        "        \n",
        "        # Modification de la tête de classification\n",
        "        in_features = self.backbone.fc.in_features\n",
        "        self.backbone.fc = nn.Identity() # On retire la couche FC d'origine\n",
        "        \n",
        "        self.classifier = nn.Sequential(\n",
        "            nn.Linear(in_features, 256),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.4),\n",
        "            nn.Linear(256, num_classes)\n",
        "        )\n",
        "        \n",
        "    def forward(self, x):\n",
        "        features = self.backbone(x)\n",
        "        logits = self.classifier(features)\n",
        "        return logits\n"
    ]

# 4. Update Optimizer and Scheduler (use AdamW)
opt_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'optimizer = optim.Adam' in "".join(cell['source']):
        opt_cell_idx = i
        break

if opt_cell_idx != -1:
    nb['cells'][opt_cell_idx]['source'] = [
        "# Optimizer AdamW avec Weight Decay pour une meilleure généralisation\n",
        "optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)\n",
        "\n",
        "# Scheduler plus réactif pour le Transfer Learning\n",
        "scheduler = optim.lr_scheduler.ReduceLROnPlateau(\n",
        "    optimizer, mode='max', factor=0.5, patience=2\n",
        ")\n",
        "\n",
        "print(\"Optimizer : AdamW configuré pour la performance\")\n",
        "print(\"Scheduler : ReduceLROnPlateau (basé sur l'AUC de validation)\")\n"
    ]

# 5. Fix Scheduler step in training loop
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'scheduler.step(val_loss)' in "".join(cell['source']):
        cell['source'] = [line.replace('scheduler.step(val_loss)', 'scheduler.step(val_auc)') for line in cell['source']]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Optimisation de SimpleCNN terminée.")
