import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Tuple, List

def make_dataset(prime_P: int = 113) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the modular addition dataset.
    Inputs: [i, j, prime_P] (where prime_P acts as the '=' token)
    Labels: (i + j) % prime_P
    """
    dataset = []
    labels = []
    
    # The notebook uses 113 as the special token for '=' which happens to be prime_P
    # We will stick to that convention as per the notebook logic
    eq_token = prime_P 
    
    for i in range(prime_P):
        for j in range(prime_P):
            label = (i + j) % prime_P
            
            # Input sequence: i, j, =
            temp_arr = [i, j, eq_token]
            
            dataset.append(torch.tensor(temp_arr, dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))
            
    main_dataset = torch.stack(dataset, dim=0)
    main_labels = torch.stack(labels, dim=0)
    
    return main_dataset, main_labels

def get_dataloaders(
    inputs: torch.Tensor, 
    labels: torch.Tensor, 
    batch_size_ratio: float, 
    seed: int = 999
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into training and validation sets and returns DataLoaders.
    """
    # Set seed for reproducibility of the split
    torch.manual_seed(seed)
    
    data = TensorDataset(inputs, labels)
    
    train_size = int(batch_size_ratio * len(data))
    val_size = len(data) - train_size
    
    train_ds, val_ds = random_split(data, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=train_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_size, shuffle=False)
    
    return train_loader, val_loader
