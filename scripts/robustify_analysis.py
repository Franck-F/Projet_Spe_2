import json
import os

notebook_path = r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\modelisation_SimpleCNN_patchs_5000.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Global replacement strategy for robustness:
# We want to use df_test_res instead of separate lists in analysis cells

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # Don't touch the training loop or inference loop where these are populated
        if 'tqdm(loader' in source_str or 'tqdm(test_loader)' in source_str:
            continue
            
        # Analysis cells
        if 'confusion_matrix(' in source_str or 'roc_curve(' in source_str or 'precision_recall_curve(' in source_str or 'average_precision_score(' in source_str:
            new_source = []
            for line in cell['source']:
                line = line.replace('all_labels', "df_test_res['label']")
                line = line.replace('all_probs', "df_test_res['prob']")
                line = line.replace('all_preds', "df_test_res['pred']")
                new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Sécurisation des variables d'analyse terminée.")
