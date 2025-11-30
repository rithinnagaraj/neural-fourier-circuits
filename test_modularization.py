import sys
import os
sys.path.append(os.path.abspath('src'))

from config import GrokkingConfig
from data import make_dataset, get_dataloaders
from model import get_model
from train import train_model
import torch

def test():
    cfg = GrokkingConfig()
    cfg.num_epochs = 2 # Run 2 epochs to trigger validation logic (validation is every 500 epochs, so I need to tweak logic or just trust it runs)
    # Actually, validation runs if (epoch+1) % 500 == 0. So 2 epochs won't trigger validation print.
    # But it will verify training loop runs.
    
    print("Generating dataset...")
    inputs, labels = make_dataset(cfg.prime_P)
    
    print("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(inputs, labels, cfg.batch_size_ratio, cfg.seed)
    
    print("Initializing model...")
    model = get_model(cfg)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Starting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=cfg.num_epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=device,
        target_val_acc=cfg.target_val_acc,
        checkpoint_dir="checkpoints"
    )

    print("Evaluating model...")
    evaluate_model(model)
    print("Test completed successfully!")

if __name__ == "__main__":
    test()
