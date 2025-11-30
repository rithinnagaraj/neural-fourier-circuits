import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from typing import Optional, List
import os

def train_model(
    model: HookedTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    target_val_acc: float = 0.975,
    checkpoint_dir: str = "../checkpoints"
) -> HookedTransformer:
    """
    Trains the model and returns the final model.
    """
    model.to(device)
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_inputs, val_labels = next(iter(val_loader))
    val_inputs.to(device)
    val_labels.to(device)
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):

        logits, labels = next(iter(train_loader))
        logits.to(device)
        labels.to(device)

        outputs = model(logits)
        final_output_logits = outputs[:, -1, :]
        loss = loss_fn(final_output_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(torch.argmax(final_output_logits, dim=-1) == labels) / final_output_logits.size(0)
        
        # Validation
        if (epoch + 1) % 500 == 0:
            val_outputs = model(val_inputs)
            val_final_output_logits = val_outputs[:, -1, :]
            val_loss = loss_fn(val_final_output_logits, val_labels)

            val_acc = torch.sum(torch.argmax(val_final_output_logits, dim=-1) == val_labels) / val_final_output_logits.size(0)

            print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Train Acc: {train_acc.item()}, Val Acc: {val_acc.item()}")
            
            if val_acc.item() > target_val_acc:
                print(f"Target validation accuracy {target_val_acc} reached. Stopping early.")
                save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
                break

    return model