import json
import os

notebook_path = r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\modelisation_SimpleCNN_patchs_5000.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Find the inference cell and add df_test_res creation
# 2. Find the old df_test_res creation and remove it to avoid redundancy

found_inference = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'Inférence sur le Test Set' in "".join(cell['source']):
        source = cell['source']
        if 'df_test_res = ' not in "".join(source):
            # Insert at the end of the cell
            source.append("\n")
            source.append("# Compilation des résultats pour analyse\n")
            source.append("df_test_res = df_test.copy().reset_index(drop=True)\n")
            source.append("df_test_res['pred'] = all_preds\n")
            source.append("df_test_res['prob'] = all_probs\n")
            source.append("df_test_res['label'] = all_labels\n")
            source.append("print(\"Compilation des résultats terminée.\")\n")
            cell['source'] = source
            found_inference = True
        break

# If inference cell with tqdm label not found, try by content
if not found_inference:
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and 'all_probs.extend(probs)' in "".join(cell['source']):
             source = cell['source']
             if 'df_test_res = ' not in "".join(source):
                source.append("\n")
                source.append("# Compilation des résultats pour analyse\n")
                source.append("df_test_res = df_test.copy().reset_index(drop=True)\n")
                source.append("df_test_res['pred'] = all_preds\n")
                source.append("df_test_res['prob'] = all_probs\n")
                source.append("df_test_res['label'] = all_labels\n")
                source.append("print(\"Compilation des résultats terminée.\")\n")
                cell['source'] = source
                found_inference = True
             break

# Clean up the old definition site if it exists elsewhere
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'df_test_res = df_test.copy().reset_index(drop=True)' in "".join(cell['source']) and len(cell['source']) < 15:
        # It's likely the old small cell, we can simplify it or comment it out
        cell['source'] = ["# Résultats déjà compilés plus haut\n", "display(df_test_res.head())\n"]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Consolidation de 'df_test_res' terminée.")
