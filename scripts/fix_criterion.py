import json
import os

notebook_path = r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\modelisation_SimpleCNN_patchs_5000.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell containing "# Fonction de Perte : Focal Loss vs BCE"
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Optimizer : AdamW' in "".join(cell['source']):
        # Inclusion of Loss definition BEFORE the Optimizer
        new_source = [
            "# --- CONFIGURATION DE LA FONCTION DE PERTE [RÉTABLI] ---\n",
            "USE_FOCAL_LOSS = True\n",
            "\n",
            "if USE_FOCAL_LOSS:\n",
            "    class FocalLoss(nn.Module):\n",
            "        def __init__(self, alpha=0.25, gamma=2):\n",
            "            super(FocalLoss, self).__init__()\n",
            "            self.alpha = alpha\n",
            "            self.gamma = gamma\n",
            "            \n",
            "        def forward(self, inputs, targets):\n",
            "            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')\n",
            "            pt = torch.exp(-BCE_loss)\n",
            "            F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss\n",
            "            return torch.mean(F_loss)\n",
            "    \n",
            "    criterion = FocalLoss(alpha=0.4, gamma=2) # Alpha ajusté pour le déséquilibre\n",
            "    print(\"Loss : Focal Loss activée\")\n",
            "else:\n",
            "    # BCE avec poids pour gérer le déséquilibre si Focal Loss est désactivée\n",
            "    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(DEVICE)\n",
            "    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n",
            "    print(\"Loss : BCE with Logits (avec pos_weight)\")\n",
            "\n",
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
        cell['source'] = new_source
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Définition de 'criterion' rétablie avec succès.")
else:
    print("Erreur: Cellule cible non trouvée.")
