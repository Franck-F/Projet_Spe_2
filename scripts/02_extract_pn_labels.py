"""
Script pour extraire les labels pN (stade patient) et créer un fichier de mapping.
Utilise les métadonnées de WILDS et les labels pN originaux de CAMELYON17.
"""
import pandas as pd
from wilds import get_dataset
from pathlib import Path
import sys

# Ajouter le dossier racine au path pour importer les modules
sys.path.append('.')

def extract_pn_labels():
    print("=== Extraction des Labels pN ===")
    
    # 1. Charger le dataset WILDS (juste les métadonnées)
    print("Chargement des métadonnées WILDS...")
    try:
        dataset = get_dataset(
            dataset="camelyon17",
            download=False,
            root_dir='data/raw/wilds'
        )
    except Exception as e:
        print(f"Erreur: Impossible de charger le dataset WILDS. Est-il téléchargé ?\n{e}")
        return

    metadata_df = dataset.metadata_df
    print(f"Métadonnées chargées : {len(metadata_df)} patchs")

    # 2. Charger le fichier example.csv (Labels officiels)
    labels_path = 'data/raw/metadata/camelyon17/example.csv'
    if not Path(labels_path).exists():
        print(f"Fichier {labels_path} introuvable.")
        print("Veuillez exécuter scripts/download_metadata.py d'abord (ou le script qui a téléchargé example.csv)")
        return
        
    print(f"Chargement des labels depuis {labels_path}...")
    official_labels = pd.read_csv(labels_path)
    
    # Nettoyer les noms de fichiers pour matcher (patient_XXX.zip -> patient_XXX)
    official_labels['patient_id'] = official_labels['patient'].astype(str).str.replace('.zip', '')
    
    # Créer un dictionnaire patient_id -> pN_stage
    # Filtrer pour ne garder que les lignes 'patient_XXX.zip' qui contiennent le stage
    patient_stages = official_labels[official_labels['patient'].str.contains('.zip')].copy()
    patient_to_stage = dict(zip(patient_stages['patient_id'], patient_stages['stage']))
    
    print(f"Labels officiels trouvés pour {len(patient_to_stage)} patients (Test set)")

    # 3. Mapper les slides WILDS vers les stages
    # Dans WILDS, 'slide' est un ID numérique (int), pas le nom 'patient_XXX_node_Y'
    # Il faut voir si on peut récupérer le mapping ID -> Nom original
    
    # Heureusement, le dataset WILDS CAMELYON17 conserve une certaine structure.
    # Mais le metadata_df a une colonne 'patient' (int) qui est l'ID du patient.
    
    # Nous devons faire le lien entre 'patient' (int) de WILDS et 'patient_XXX' officiel.
    # Pour le training set (hôpitaux 0, 1, 2), les pN stages sont connus et étaient dans training/annotations.
    # Pour le test set (hôpitaux 3, 4), c'est là où example.csv est utile (patients 100-199).
    
    # Créons un DataFrame final par patient
    unique_patients = metadata_df[['patient', 'hospital']].drop_duplicates()
    unique_patients['pn_stage'] = 'Unknown'
    
    # Mapping manuel approximatif basé sur les ranges d'ID si nécessaire, 
    # mais essayons d'abord de voir si on peut corréler.
    
    # NOTE: WILDS a déjà splité Train/Val/Test.
    # Le 'test' split de WILDS correspond aux données 'Test' de CAMELYON17 (patients 100-199 de example.csv).
    # Vérifions cela.
    
    print("\nAnalyse des patients par split...")
    for split in ['train', 'val', 'test']:
        split_mask = dataset.split_dict[split]
        split_patients = metadata_df.iloc[split_mask]['patient'].unique()
        print(f"Split {split}: {len(split_patients)} patients (IDs: {split_patients.min()}-{split_patients.max()})")

    # On va supposer que pour le split 'test', les IDs correspondent aux patients du fichier example.csv
    # Les patients dans example.csv sont 100 à 199.
    # Vérifions si les IDs WILDS matchent.
    
    # Création du fichier de sortie
    output_path = 'data/processed/patient_pn_labels.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # On va sauvegarder ce qu'on peut
    # Pour l'instant, on sauvegarde le mapping connu du example.csv
    patient_stages[['patient_id', 'stage']].to_csv(output_path, index=False)
    print(f"\nMapping Patient -> Stage sauvegardé dans {output_path}")
    print("Contenu (5 premiers) :")
    print(patient_stages[['patient_id', 'stage']].head())
    
    # Note: Pour le training, nous devrons peut-être reconstruire les labels pN 
    # à partir des annotations XML si WILDS ne les donne pas explicitement.
    # Mais WILDS donne 'y' (patch label).
    # Le pN stage pour le training est implicite (calculable ou fourni ailleurs).
    
    # Vérifions si on peut déduire le pN stage du training set.
    # (Ce sera une étape suivante si nécessaire).

if __name__ == "__main__":
    extract_pn_labels()
