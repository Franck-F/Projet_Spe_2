import json
import os

notebook_path = r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\notebooks\modelisation_SimpleCNN_patchs_5000.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the Grad-CAM visualization cell
# We look for the cell containing "target_layer = model.conv3"
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'target_layer = model.conv3' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'target_layer = model.layer4[-1].conv2' in line:
                new_source.append(line.replace('model.layer4[-1].conv2', 'model.backbone.layer4[-1]'))
            elif 'target_layer = model.conv3' in line:
                # Replace the whole logic to be more specific to our new SimpleCNN
                new_source.append('    target_layer = model.backbone.layer4[-1]\n')
                # Skip the next few lines if they are part of the old if/else
            elif 'if USE_RESNET:' in line or 'else:' in line or 'target_layer = model.conv3' in line:
                continue
            else:
                new_source.append(line)
        cell['source'] = new_source
        found = True
        break

# If the above replacement was too complex, let's try a simpler one: just replace the specific lines
if not found:
    for cell in nb['cells']:
         if cell['cell_type'] == 'code' and 'Grad-CAM' in "".join(cell['source']) and 'target_layer' in "".join(cell['source']):
            new_source = []
            skip_next = False
            for line in cell['source']:
                if 'if USE_RESNET:' in line:
                    new_source.append('    # Ciblage de la couche pour la nouvelle architecture SimpleCNN (ResNet-18 backbone)\n')
                    new_source.append('    target_layer = model.backbone.layer4[-1]\n')
                    skip_next = True
                elif skip_next:
                    if 'grad_cam =' in line:
                        new_source.append(line)
                        skip_next = False
                    continue
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Correction Grad-CAM terminée.")
