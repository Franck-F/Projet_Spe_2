import json
from pathlib import Path

notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\03_modeling_patch.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to find cell index by source content
def find_cell_index_contains(source_snippet):
    for i, cell in enumerate(nb['cells']):
        if source_snippet in "".join(cell['source']):
            return i
    return -1

# 1. Locate insertion point: Before "Évaluation et Interprétabilité"
idx_eval_header = find_cell_index_contains("# Évaluation et Interprétabilité")
idx_metrics_code = find_cell_index_contains("# Inference sur le Test Set")

if idx_eval_header != -1 and idx_metrics_code != -1:
    # 2. Extract the inference code from the metrics cell
    metrics_source = nb['cells'][idx_metrics_code]['source']
    
    # Split source into Inference part and Metrics part
    inference_part = [
        "from tqdm.auto import tqdm\n", # Fix NameError
        "\n",
        "# Chargement du meilleur modèle\n",
        "try:\n",
        "    model.load_state_dict(torch.load('../models/baseline_v1/best_model.pth'))\n",
        "    print(\"Meilleur modèle chargé.\")\n",
        "except FileNotFoundError:\n",
        "    print(\"Attention: Modèle non trouvé, on utilise le modèle courant.\")\n",
        "\n",
        "model.eval()\n",
        "model.to(DEVICE)\n",
        "\n",
        "# Inference sur le Test Set\n",
        "all_probs = []\n",
        "all_preds = []\n",
        "all_labels = []\n",
        "\n",
        "print(\"Inference sur le Test Set...\")\n",
        "with torch.no_grad():\n",
        "    for batch in tqdm(test_loader):\n",
        "        if len(batch) == 3:\n",
        "            images, labels, _ = batch\n",
        "        else:\n",
        "            images, labels = batch\n",
        "        \n",
        "        images = images.to(DEVICE)\n",
        "        \n",
        "        outputs = model(images)\n",
        "        probs = torch.sigmoid(outputs).cpu().numpy().flatten()\n",
        "        preds = (probs > 0.5).astype(int)\n",
        "        \n",
        "        all_probs.extend(probs)\n",
        "        all_preds.extend(preds)\n",
        "        all_labels.extend(labels.numpy().flatten())\n"
    ]
    
    metrics_only_part = [
        "# Métriques Globales\n",
        "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "print(classification_report(all_labels, all_preds, target_names=['Normal', 'Tumor']))\n",
        "auc = roc_auc_score(all_labels, all_probs)\n",
        "print(f\"AUC-ROC Global: {auc:.4f}\")\n",
        "\n",
        "# Matrice de Confusion\n",
        "cm = confusion_matrix(all_labels, all_preds)\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
        "plt.title('Matrice de Confusion (Patch-Level)')\n",
        "plt.show()"
    ]
    
    # 3. Create new Test Section cells
    test_header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Test"]
    }
    
    test_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": inference_part
    }
    
    # 4. Insert before Eval Header
    nb['cells'].insert(idx_eval_header, test_code_cell)
    nb['cells'].insert(idx_eval_header, test_header_cell)
    
    # 5. Update the existing Metrics cell to only contain metrics
    # Note: Indicies shifted by +2
    nb['cells'][idx_metrics_code + 2]['source'] = metrics_only_part

    # 6. Update the Eval Header to be Section 6
    nb['cells'][idx_eval_header + 2]['source'] = ["# 6. Évaluation et Interprétabilité\n", "\n", "## Objectifs\n..."]

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook updated: Test section inserted and NameError fixed.")
else:
    print("Could not find insertion points.")
