"""
src/train.py - Training pipeline for 4-Class Rice Field Neural Network on GPU (Dry, Flooded, Planted, Others).
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import build_classifier, CLASSES
from src.dataset import create_dataloaders


def train_classifier(
    dataset_dir="Dataset",
    epochs=25,
    batch_size=16,
    learning_rate=1e-3,
    save_path="rice_field_classifier.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[*] GPU Model: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[*] Target Classes ({len(CLASSES)}): {CLASSES}", flush=True)
    
    # 1. Dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_train_samples=800,
        num_val_samples=160
    ) 
    
    # 2. Model
    model = build_classifier(num_classes=len(CLASSES)).to(device)
    
    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    best_model_state = None
    start_time = time.time()
    
    print("\n" + "="*75, flush=True)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^10} | {'LR':^10} | {'Time':^8}", flush=True)
    print("="*75, flush=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        class_correct = [0] * len(CLASSES)
        class_total = [0] * len(CLASSES)
        
        with torch.no_grad():
            for val_img, val_lbl in val_loader:
                val_img = val_img.to(device)
                val_lbl = val_lbl.to(device)
                
                v_out = model(val_img)
                v_loss = criterion(v_out, val_lbl)
                total_val_loss += v_loss.item()
                
                v_preds = torch.argmax(v_out, dim=1)
                val_correct += (v_preds == val_lbl).sum().item()
                val_total += val_lbl.size(0)
                
                for p, t in zip(v_preds, val_lbl):
                    if p == t:
                        class_correct[t.item()] += 1
                    class_total[t.item()] += 1
                    
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100.0
        epoch_time = time.time() - start_time
        
        print(f"{epoch:^7} | {avg_train_loss:^12.4f} | {avg_val_loss:^10.4f} | {val_acc:^9.2f}% | {current_lr:^10.6f} | {epoch_time:^7.1f}s", flush=True)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            
    print("="*75, flush=True)
    print(f"[+] Training completed! Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    
    # Save checkpoint
    checkpoint_data = {
        "model_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
        "classes": CLASSES,
        "val_accuracy": best_val_acc,
        "epoch": epochs
    }
    torch.save(checkpoint_data, save_path)
    print(f"[+] Saved model checkpoint to: {save_path}", flush=True)
    
    print("\n[+] Final Per-Class Accuracy Breakdown:", flush=True)
    for idx, cname in enumerate(CLASSES):
        c_acc = (class_correct[idx] / max(1, class_total[idx])) * 100.0
        print(f"   - {cname:10s}: {c_acc:5.1f}% ({class_correct[idx]}/{class_total[idx]})", flush=True)
    print()
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rice Field 4-Class Neural Network on GPU.")
    parser.add_argument("--dataset", type=str, default="Dataset", help="Path to Dataset directory")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    train_classifier(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
