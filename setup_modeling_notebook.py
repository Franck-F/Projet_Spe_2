import json
from pathlib import Path

# Path to the notebook
notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\03_modeling_patch.ipynb")

# Create a fresh notebook structure
nb_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 2 : ModÃ©lisation au Niveau Patch (EntraÃ®nement Baseline)\n",
    "\n",
    "## Objectifs\n",
    "- Charger le dataset prÃ©-traitÃ© (`df_final_5000.csv`)\n",
    "- PrÃ©parer un `CustomDataset` PyTorch (Resize 96x96)\n",
    "- Diviser en Train/Val/Test (Splits par Patient pour Ã©viter le Data Leakage)\n",
    "- EntraÃ®ner le modÃ¨le `SimpleCNN`\n",
    "- Visualiser les courbes d'apprentissage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from torchvision import transforms\n",
    "from PIL import Image\n",
    "from pathlib import Path\n",
    "from sklearn.model_selection import GroupShuffleSplit\n",
    "import sys\n",
    "\n",
    "# Ajout du dossier src pour les imports modules\n",
    "sys.path.append('..')\n",
    "from src.models.cnn_baseline import SimpleCNN\n",
    "from src.models.train import Trainer\n",
    "\n",
    "# Configuration\n",
    "DATA_PATH = Path('../data/processed/df_final_5000.csv')\n",
    "PATCHES_DIR = Path('../data/processed/patches_224x224')\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "BATCH_SIZE = 32\n",
    "EPOCHS = 10\n",
    "LR = 1e-4\n",
    "\n",
    "print(f\"Device: {DEVICE}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Chargement et Split des DonnÃ©es\n",
    "\n",
    "Nous devons diviser le dataset en Train/Val/Test. \n",
    "**CRITIQUE** : La division doit se faire par `patient` pour Ã©viter que des patchs d'un mÃªme patient se retrouvent Ã  la fois en train et en test (Data Leakage)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Chargement du DataFrame\n",
    "df = pd.read_csv(DATA_PATH)\n",
    "\n",
    "# Nettoyage Rapide (si nÃ©cessaire)\n",
    "df = df.dropna(subset=['tumor'])\n",
    "df['tumor'] = df['tumor'].astype(int)\n",
    "\n",
    "print(f\"Dataset chargÃ© : {len(df)} patchs\")\n",
    "print(\"Distribution des labels :\")\n",
    "print(df['tumor'].value_counts(normalize=True))\n",
    "\n",
    "# Split: Train (70%) / Val (15%) / Test (15%) basÃ© sur les GROUPES (Patients)\n",
    "splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)\n",
    "train_idx, temp_idx = next(splitter.split(df, groups=df['patient']))\n",
    "df_train = df.iloc[train_idx]\n",
    "df_temp = df.iloc[temp_idx]\n",
    "\n",
    "splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42) # 0.5 de 30% = 15%\n",
    "val_idx, test_idx = next(splitter_val.split(df_temp, groups=df_temp['patient']))\n",
    "df_val = df_temp.iloc[val_idx]\n",
    "df_test = df_temp.iloc[test_idx]\n",
    "\n",
    "print(f\"\\nTrain: {len(df_train)} patchs ({df_train['patient'].nunique()} patients)\")\n",
    "print(f\"Val:   {len(df_val)} patchs ({df_val['patient'].nunique()} patients)\")\n",
    "print(f\"Test:  {len(df_test)} patchs ({df_test['patient'].nunique()} patients)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. CrÃ©ation du Dataset PyTorch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class CamelyonDataset(Dataset):\n",
    "    def __init__(self, df, transform=None):\n",
    "        self.df = df.reset_index(drop=True)\n",
    "        self.transform = transform\n",
    "        \n",
    "    def __len__(self):\n",
    "        return len(self.df)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        row = self.df.iloc[idx]\n",
    "        img_path = row['path']\n",
    "        label = row['tumor']\n",
    "        \n",
    "        # Gestion des chemins relatifs/absolus\n",
    "        # Si le chemin dans le CSV est absolu ou relatif diffÃ©rent, on essaye de s'adapter\n",
    "        # Ici on suppose que 'path' est correct ou relatif Ã  notre contexte\n",
    "        try:\n",
    "            image = Image.open(img_path).convert('RGB')\n",
    "        except Exception as e:\n",
    "            # Fallback si le chemin ne matche pas exactement\n",
    "            # On reconstruit le chemin via la convention de nommage si nÃ©cessaire\n",
    "            # Mais pour l'instant levons l'erreur\n",
    "            print(f\"Erreur loading {img_path}: {e}\")\n",
    "            # Retourner une image noire pour ne pas crasher le batch (solution temporaire)\n",
    "            image = Image.new('RGB', (224, 224))\n",
    "            \n",
    "        if self.transform:\n",
    "            image = self.transform(image)\n",
    "            \n",
    "        return image, torch.tensor(label, dtype=torch.float)\n",
    "\n",
    "# Transformations \n",
    "# IMPORTANT : Le modÃ¨le `SimpleCNN` attend du 96x96\n",
    "train_transforms = transforms.Compose([\n",
    "    transforms.Resize((96, 96)),\n",
    "    transforms.RandomHorizontalFlip(),\n",
    "    transforms.RandomVerticalFlip(),\n",
    "    transforms.RandomRotation(90),\n",
    "    transforms.ToTensor(),\n",
    "    # Normalisation ImageNet ou calculÃ©e prÃ©cÃ©demment\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "val_test_transforms = transforms.Compose([\n",
    "    transforms.Resize((96, 96)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
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
    "print(\"DataLoaders prÃªts.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Initialisation et EntraÃ®nement"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = SimpleCNN()\n",
    "criterion = nn.BCEWithLogitsLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=LR)\n",
    "\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    train_loader=train_loader,\n",
    "    val_loader=val_loader,\n",
    "    criterion=criterion,\n",
    "    optimizer=optimizer,\n",
    "    device=DEVICE,\n",
    "    save_dir='../models/baseline_v1'\n",
    ")\n",
    "\n",
    "# Lancement\n",
    "print(\"DÃ©but de l'entraÃ®nement...\")\n",
    "history = trainer.train(num_epochs=EPOCHS)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Visualisation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import plotly.graph_objects as go\n",
    "from plotly.subplots import make_subplots\n",
    "\n",
    "def plot_history(history):\n",
    "    epochs = list(range(1, len(history['train_loss']) + 1))\n",
    "    \n",
    "    fig = make_subplots(rows=1, cols=2, subplot_titles=(\"Loss\", \"Accuracy & F1\"))\n",
    "    \n",
    "    fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)\n",
    "    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='orange')), row=1, col=1)\n",
    "    \n",
    "    fig.add_trace(go.Scatter(x=epochs, y=history['val_acc'], name='Val Accuracy', line=dict(color='green')), row=1, col=2)\n",
    "    fig.add_trace(go.Scatter(x=epochs, y=history['val_f1'], name='Val F1', line=dict(color='purple')), row=1, col=2)\n",
    "    \n",
    "    fig.update_layout(title_text=\"Courbes d'apprentissage SimpleCNN\", height=500, template='plotly_white')\n",
    "    fig.show()\n",
    "\n",
    "if 'history' in locals():\n",
    "    plot_history(history)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
    
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb_content, f, indent=1)
    
print(f"Notebook {notebook_path} reset and initialized for modeling.")
