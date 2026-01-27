import json
from pathlib import Path

def update_notebook_path(notebook_path, old_str, new_str):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # cell['source'] can be a list of strings or a single string
            if isinstance(cell['source'], list):
                for i, line in enumerate(cell['source']):
                    if old_str in line:
                        cell['source'][i] = line.replace(old_str, new_str)
                        modified = True
            elif isinstance(cell['source'], str):
                if old_str in cell['source']:
                    cell['source'] = cell['source'].replace(old_str, new_str)
                    modified = True
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {notebook_path}")
    else:
        print(f"No changes made to {notebook_path}")

nb_path = Path(r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\modele_Stade.ipynb')
update_notebook_path(nb_path, "df_20000.csv", "df_full_normalized.csv")
