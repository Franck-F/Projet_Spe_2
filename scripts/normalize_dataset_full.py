import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

def stain_normalization_reinhard(img, reference_mean, reference_std):
    """Méthode Reinhard : Normalisation basée sur moyenne/variance RGB."""
    img = np.array(img, dtype=np.float32)
    
    # Stats actuelles
    current_mean = np.mean(img, axis=(0, 1))
    current_std = np.std(img, axis=(0, 1))
    
    # Normalisation
    img_norm = np.zeros_like(img)
    for i in range(3):
        if current_std[i] > 0:
            img_norm[:, :, i] = (img[:, :, i] - current_mean[i]) * (reference_std[i] / current_std[i]) + reference_mean[i]
        else:
            img_norm[:, :, i] = img[:, :, i]
    
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    return img_norm

# Global variables for workers
_tpl_mean = None
_tpl_std = None

def init_worker(mean, std):
    global _tpl_mean, _tpl_std
    _tpl_mean = mean
    _tpl_std = std

def process_one_patch(args):
    src_path, dst_path = args
    if dst_path.exists():
        return True
    
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        
        # BGR -> RGB for normalization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_norm = stain_normalization_reinhard(img_rgb, _tpl_mean, _tpl_std)
        
        # RGB -> BGR for saving
        img_final = cv2.cvtColor(img_norm, cv2.COLOR_RGB2BGR)
        
        # Ensure directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        cv2.imwrite(str(dst_path), img_final, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return True
    except Exception:
        return False

def main():
    SRC_DIR = Path(r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\data\processed\patches_224x224')
    DST_DIR = Path(r'c:\Users\Franck\.gemini\antigravity\scratch\Projet_Spe_2\data\processed\patches_224x224_normalized')
    
    print(f"Listing files in {SRC_DIR}...")
    all_files = list(SRC_DIR.rglob('*.png'))
    print(f"Found {len(all_files)} patches.")
    
    # 1. Calculate Template
    print("Calculating global template from 10,000 random patches...")
    sample_files = random.sample(all_files, min(10000, len(all_files)))
    
    means = []
    stds = []
    for f in tqdm(sample_files, desc="Sampling"):
        img = cv2.imread(str(f))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            means.append(np.mean(img_rgb, axis=(0, 1)))
            stds.append(np.std(img_rgb, axis=(0, 1)))
            
    template_mean = np.median(means, axis=0)
    template_std = np.median(stds, axis=0)
    
    print(f"Template Mean: {template_mean}")
    print(f"Template Std:  {template_std}")
    
    # 2. Process all
    tasks = []
    for src_path in all_files:
        rel_path = src_path.relative_to(SRC_DIR)
        dst_path = DST_DIR / rel_path
        tasks.append((src_path, dst_path))
        
    print(f"Starting optimized normalization of {len(tasks)} patches...")
    
    num_workers = min(os.cpu_count(), 8)
    
    # Using initializer and chunksize for better performance on Windows
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(template_mean, template_std)) as executor:
        # We use a large chunksize to reduce pickling overhead between processes
        list(tqdm(executor.map(process_one_patch, tasks, chunksize=100), total=len(tasks), desc="Normalizing"))

if __name__ == "__main__":
    main()
