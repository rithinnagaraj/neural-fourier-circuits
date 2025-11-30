import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from transformer_lens import HookedTransformer

def evaluate_model(model: HookedTransformer) -> None:
    """
    Evaluates the model and saves the graphs.
    """
    os.makedirs("plots", exist_ok=True)

    # Fourier Spectrum of Embeddings
    W_E = model.W_E.detach().cpu()
    W_E_numbers = W_E[:-1, :] 

    fft_result = torch.fft.fft(W_E_numbers, dim=0)
    fft_magnitude = torch.abs(fft_result)

    half_n = fft_magnitude.shape[0] // 2
    fft_viz = fft_magnitude[:half_n, :]

    plt.figure(figsize=(12, 6))
    plt.imshow(fft_viz.T, aspect='auto', cmap='inferno', origin='lower')

    plt.title("The Signature of Grokking: Fourier Spectrum of Embeddings", fontsize=14)
    plt.xlabel("Frequency Component (Periodicity)", fontsize=12)
    plt.ylabel("Neuron Dimension (0-128)", fontsize=12)
    plt.colorbar(label="Magnitude (Importance)")
    plt.tight_layout()

    save_path = "plots/fourier_spectrum.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")

    # Attention Oscillations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    queries = torch.zeros((113, 3), dtype=torch.long).to(device)
    queries[:, 0] = 0
    queries[:, 1] = torch.arange(113).to(device)
    queries[:, 2] = 113

    with torch.no_grad():
        logits, cache = model.run_with_cache(queries)

    attn = cache['pattern', 0, 'attn']
    attention_to_b = attn[:, :, 2, 1].cpu()

    plt.figure(figsize=(12, 5))
    for head_idx in range(4):
        plt.plot(attention_to_b[:, head_idx], label=f"Head {head_idx}", alpha=0.8)

    plt.title("Attention Oscillations: Does the model treat numbers as waves?", fontsize=14)
    plt.xlabel("Input Number 'b' (0-112)", fontsize=12)
    plt.ylabel("Attention Score", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = "plots/attention_oscillations.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")

    plt.show()
