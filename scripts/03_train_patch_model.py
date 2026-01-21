"""
Script d'entraînement pour le modèle patch-level sur CAMELYON17 (WILDS)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm

from src.data.wilds_loader import get_wilds_dataloaders
from src.models.cnn_baseline import BaselineCNN
from src.utils.config import get_device

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Les dataloaders WILDS retournent un dict
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{desc}]")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
            
    return running_loss / len(loader), 100. * correct / total

def main():
    # Configuration
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-4
    DEVICE = get_device()
    OUTPUT_DIR = Path('models/checkpoints')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print("Initialisation des DataLoaders...")
    train_loader, val_loader, test_loader = get_wilds_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4
    )
    
    print("Initialisation du Modèle...")
    model = BaselineCNN(num_classes=2).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Historique
    history = []
    
    print("\n=== Début de l'entraînement ===")
    start_time = time.time()
    
    best_val_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, desc="Val")
        
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })
        
        # Sauvegarde du meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pth')
            print(f"✅ Nouveau meilleur modèle sauvegardé (Acc: {val_acc:.2f}%)")
            
    total_time = time.time() - start_time
    print(f"\nEntraînement terminé en {total_time/60:.1f} minutes")
    
    # Évaluation finale sur Test set
    print("\nÉvaluation sur le Test Set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE, desc="Test")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Sauvegarde de l'historique
    pd.DataFrame(history).to_csv('results/metrics/training_history.csv', index=False)
    print("Historique sauvegardé dans results/metrics/training_history.csv")

if __name__ == "__main__":
    main()
