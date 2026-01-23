import json
from pathlib import Path

notebook_path = Path(r"C:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\03_modeling_patch.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to find cell index by source content
def find_cell_index(source_snippet):
    for i, cell in enumerate(nb['cells']):
        if source_snippet in "".join(cell['source']):
            return i
    return -1

# 1. Implement Metrics (Section 1)
idx_metrics = find_cell_index("# TODO: Calculer toutes les métriques")
if idx_metrics != -1:
    nb['cells'][idx_metrics] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Chargement du meilleur modèle\n",
            "model.load_state_dict(torch.load('../models/baseline_v1/best_model.pth'))\n",
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
            "        # labels = labels.to(DEVICE).float() # Pas besoin des labels pour predict mais bon\n",
            "        \n",
            "        outputs = model(images)\n",
            "        probs = torch.sigmoid(outputs).cpu().numpy().flatten()\n",
            "        preds = (probs > 0.5).astype(int)\n",
            "        \n",
            "        all_probs.extend(probs)\n",
            "        all_preds.extend(preds)\n",
            "        all_labels.extend(labels.numpy().flatten())\n",
            "\n",
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
    }

# 2. Implement Domain Shift (Section 2)
idx_domain = find_cell_index("# TODO: Performances par hôpital")
if idx_domain != -1:
    nb['cells'][idx_domain] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyse par Centre (Domain Shift)\n",
            "# On ajoute les prédictions au DataFrame Test\n",
            "df_test_res = df_test.copy()\n",
            "df_test_res['pred'] = all_preds\n",
            "df_test_res['prob'] = all_probs\n",
            "df_test_res['label'] = all_labels\n",
            "\n",
            "# Calcul de l'Accuracy par centre\n",
            "center_metrics = df_test_res.groupby('center').apply(\n",
            "    lambda x: pd.Series({\n",
            "        'accuracy': (x['pred'] == x['label']).mean(),\n",
            "        'tumor_prevalence': x['label'].mean(),\n",
            "        'count': len(x)\n",
            "    })\n",
            ").reset_index()\n",
            "\n",
            "print(\"Performances par Centre :\")\n",
            "display(center_metrics)\n",
            "\n",
            "# Visualisation\n",
            "import plotly.express as px\n",
            "fig = px.bar(center_metrics, x='center', y='accuracy', \n",
            "             color='center', title='Accuracy par Centre (Evaluation Domain Shift)')\n",
            "fig.add_hline(y=df_test_res['pred'].eq(df_test_res['label']).mean(), line_dash=\"dash\", annotation_text=\"Global Acc\")\n",
            "fig.show()"
        ]
    }

# 3. Implement Grad-CAM (Section 3)
idx_gradcam = find_cell_index("# TODO: Grad-CAM sur cas représentatifs")
if idx_gradcam != -1:
    nb['cells'][idx_gradcam] = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Implémentation Grad-CAM Simple (Hooks)\n",
            "class GradCAM:\n",
            "    def __init__(self, model, target_layer):\n",
            "        self.model = model\n",
            "        self.target_layer = target_layer\n",
            "        self.gradients = None\n",
            "        self.activations = None\n",
            "        \n",
            "        # Hooks\n",
            "        target_layer.register_forward_hook(self.save_activation)\n",
            "        target_layer.register_full_backward_hook(self.save_gradient)\n",
            "\n",
            "    def save_activation(self, module, input, output):\n",
            "        self.activations = output\n",
            "\n",
            "    def save_gradient(self, module, grad_input, grad_output):\n",
            "        self.gradients = grad_output[0]\n",
            "\n",
            "    def __call__(self, x):\n",
            "        self.model.zero_grad()\n",
            "        output = self.model(x)\n",
            "        output.backward()\n",
            "        \n",
            "        # Pooling des gradients\n",
            "        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])\n",
            "        \n",
            "        # Pondération des activations\n",
            "        activation = self.activations[0]\n",
            "        for i in range(activation.shape[0]):\n",
            "            activation[i, :, :] *= pooled_gradients[i]\n",
            "            \n",
            "        heatmap = torch.mean(activation, dim=0).cpu().detach()\n",
            "        heatmap = np.maximum(heatmap, 0) # ReLU\n",
            "        heatmap /= torch.max(heatmap) # Normalisation\n",
            "        \n",
            "        return heatmap.numpy()\n",
            "\n",
            "# Sélection d'une image Tumorale du Test Set\n",
            "tumor_indices = df_test_res[df_test_res['label'] == 1].index\n",
            "sample_idx = tumor_indices[0] # Premier cas tumoral\n",
            "sample_img, sample_label = test_dataset[sample_idx]\n",
            "\n",
            "# Init GradCAM sur la dernière couche de convolution (conv3)\n",
            "grad_cam = GradCAM(model, model.conv3)\n",
            "\n",
            "# Génération Heatmap\n",
            "heatmap = grad_cam(sample_img.unsqueeze(0).to(DEVICE))\n",
            "\n",
            "# Visualisation Superposée\n",
            "import cv2\n",
            "img_np = sample_img.permute(1, 2, 0).numpy()\n",
            "# De-normalization si nécessaire (ici on suppose brut ou proche)\n",
            "img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())\n",
            "\n",
            "heatmap_resized = cv2.resize(heatmap, (224, 224))\n",
            "heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)\n",
            "heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0\n",
            "\n",
            "superimposed = heatmap_colored * 0.4 + img_np * 0.6\n",
            "\n",
            "plt.figure(figsize=(10, 4))\n",
            "plt.subplot(1, 3, 1); plt.imshow(img_np); plt.title('Lame Originale')\n",
            "plt.subplot(1, 3, 2); plt.imshow(heatmap); plt.title('Heatmap Grad-CAM')\n",
            "plt.subplot(1, 3, 3); plt.imshow(superimposed); plt.title('Superposition')\n",
            "plt.show()"
        ]
    }

# 4. Fill Discussion (Section 4)
idx_disc = find_cell_index("# TODO: Recommandations cliniques")
if idx_disc != -1:
    nb['cells'][idx_disc] = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Analyse des Résultats\n",
            "- **Biais Observés** : Si les performances varient fortement entre les centres (voir graphique ci-dessus), cela indique un manque de robustesse au Domain Shift.\n",
            "- **Grad-CAM** : La heatmap doit se concentrer sur les zones cellulaires denses et atypiques (noyaux larges, sombres). Si elle regarde le fond blanc ou des artefacts, le modèle biaise.\n",
            "\n",
            "### Recommandations pour la Phase 3\n",
            "1.  **Agrégation** : Utiliser le `Max Pooling` ou `Top-K` pour le diagnostic patient.\n",
            "2.  **Stratégie** : Si le biais centre est fort, envisager des techniques de *Stain Normalization* plus poussées ou du *Domain Adversarial Training*."
        ]
    }

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Evaluation sections filled successfully.")
