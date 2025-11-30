from transformer_lens import HookedTransformer, HookedTransformerConfig
from .config import GrokkingConfig

def get_model(config: GrokkingConfig) -> HookedTransformer:
    """
    Initializes the HookedTransformer model based on the provided configuration.
    """
    tl_cfg = HookedTransformerConfig(
        n_layers=config.n_layers,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_head=config.d_head,
        n_ctx=config.n_ctx,
        d_vocab=config.d_vocab,
        act_fn=config.act_fn,
        seed=config.seed,
        normalization_type=config.normalization_type
    )
    
    model = HookedTransformer(tl_cfg)
    return model
