from dataclasses import dataclass
from typing import Optional

@dataclass
class GrokkingConfig:
    # Model architecture
    n_layers: int = 1
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    n_ctx: int = 3
    d_vocab: int = 114
    act_fn: str = 'relu'
    normalization_type: Optional[str] = None
    
    # Training hyperparameters
    seed: int = 999
    prime_P: int = 113
    num_epochs: int = 20000
    lr: float = 0.001
    weight_decay: float = 1.0
    batch_size_ratio: float = 0.30
    
    # Early stopping
    target_val_acc: float = 0.975
