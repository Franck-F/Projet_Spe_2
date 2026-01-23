import json
from pathlib import Path

notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\03_modeling_patch.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Cells to append
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Ã‰valuation sur le Test Set"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Chargement du meilleur modÃ¨le\n",
            "best_model_path = '../models/baseline_v1/best_model.pth'\n",
            "model.load_state_dict(torch.load(best_model_path))\n",
            "model.eval()\n",
            "model.to(DEVICE)\n",
            "print(\"Meilleur modÃ¨le chargÃ©.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from tqdm.auto import tqdm\n",
            "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# In fÃ©rence sur le Test Set\n",
            "all_preds = []\n",
            "all_probs = []\n",
            "all_labels = []\n",
            "\n",
            "print(\"Inference sur le Test Set...\")\n",
            "with torch.no_grad():\n",
            "    for batch in tqdm(test_loader):\n",
            "        # Gestion batch dynamique (2 ou 3 elements)\n",
            "        if len(batch) == 3:\n",
            "            images, labels, _ = batch\n",
            "        else:\n",
            "            images, labels = batch\n",
            "            \n",
            "        images = images.to(DEVICE)\n",
            "        labels = labels.to(DEVICE).float()\n",
            "        \n",
            "        outputs = model(images)\n",
            "        probs = torch.sigmoid(outputs).cpu().numpy().flatten()\n",
            "        preds = (probs > 0.5).astype(int)\n",
            "        \n",
            "        all_probs.extend(probs)\n",
            "        all_preds.extend(preds)\n",
            "        all_labels.extend(labels.cpu().numpy().flatten())\n",
            "\n",
            "# MÃ©triques\n",
            "print(\"\\n--- Classification Report ---\")\n",
            "print(classification_report(all_labels, all_preds, target_names=['Normal', 'Tumor']))\n",
            "\n",
            "auc = roc_auc_score(all_labels, all_probs)\n",
            "print(f\"AUC-ROC: {auc:.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualisation : Matrice de Confusion\n",
            "cm = confusion_matrix(all_labels, all_preds)\n",
            "plt.figure(figsize=(6, 5))\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', \n",
            "            xticklabels=['Normal', 'Tumor'], yticklabels=['Normal', 'Tumor'])\n",
            "plt.ylabel('Vrai')\n",
            "plt.xlabel('PrÃ©dit')\n",
            "plt.title('Matrice de Confusion (Test Patch-Level)')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualisation : Courbe ROC\n",
            "fpr, tpr, _ = roc_curve(all_labels, all_probs)\n",
            "plt.figure(figsize=(8, 6))\n",
            "plt.plot(fpr, tpr, label=f'SimpleCNN (AUC = {auc:.3f})')\n",
            "plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)\n",
            "plt.xlabel('False Positive Rate')\n",
            "plt.ylabel('True Positive Rate')\n",
            "plt.title('Courbe ROC')\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Sauvegarde des PrÃ©dictions (pour AgrÃ©gation)\n",
            "\n",
            "Nous sauvegardons les prÃ©dictions associÃ©es aux mÃ©tadonnÃ©es (Patient, Node) pour l'Ã©tape d'agrÃ©gation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Construction du DataFrame de prÃ©dictions\n",
            "df_preds = df_test.copy().reset_index(drop=True)\n",
            "df_preds['prob_tumor'] = all_probs\n",
            "df_preds['pred_tumor'] = all_preds\n",
            "df_preds['label_true'] = all_labels\n",
            "\n",
            "save_path = Path('../data/processed/test_predictions.csv')\n",
            "df_preds.to_csv(save_path, index=False)\n",
            "print(f\"PrÃ©dictions sauvegardÃ©es dans : {save_path}\")\n",
            "display(df_preds.head())"
        ]
    }
]

nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Evaluation cells appended successfully.")
