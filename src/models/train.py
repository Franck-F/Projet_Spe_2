import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Trainer:
    """
    Classe utilitaire pour gérer l'entraînement du modèle.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_dir='models'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels, _ in pbar: # On ignore les métadonnées ici
            images = images.to(self.device)
            labels = labels.float().to(self.device).view(-1, 1) # (Batch, 1)
            
            # Zero grad
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.float().to(self.device).view(-1, 1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                # Conversion pour scikit-learn
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return epoch_loss, acc, f1

    def train(self, num_epochs=10):
        print(f"Démarrage de l'entraînement sur {self.device}...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.evaluate()
            
            # Logging
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Checkpointing (Best Loss Model)
            if val_loss < self.best_val_loss:
                print(f"✨ Amélioration (Loss: {self.best_val_loss:.4f} -> {val_loss:.4f}). Sauvegarde...")
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')
                
        total_time = time.time() - start_time
        print(f"\nEntraînement terminé en {total_time/60:.2f} minutes.")
        return self.history
