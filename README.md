# Grokking Modular Addition

This project implements a modularized version of the Grokking experiment, training a small transformer model on modular addition tasks. It is designed to be clean, reproducible, and easy to experiment with.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project requires PyTorch. If you need a specific version (e.g., for CUDA support), please install it following instructions from [pytorch.org](https://pytorch.org/) before installing other requirements.*

## Usage

### Running the Verification Script
To verify that the modularization works correctly and the model can train:
```bash
python test_modularization.py
```

### Running the Analysis Notebook
The main analysis and visualization are located in `notebooks/analysis.ipynb`.
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Modules Description

*   **`src.config`**: Defines the `GrokkingConfig` dataclass containing model parameters (layers, heads, dimensions) and training hyperparameters (learning rate, weight decay, epochs).
*   **`src.data`**: Contains `make_dataset` for generating the modular addition data ($a + b \pmod P$) and `get_dataloaders` for creating PyTorch DataLoaders.
*   **`src.model`**: Wraps `TransformerLens`'s `HookedTransformer` to create the model based on the config.
*   **`src.train`**: Implements the training loop with validation, early stopping, and checkpointing.

## About Grokking
"Grokking" refers to the phenomenon where a model initially overfits to the training data (high training accuracy, low validation accuracy) but, after a long period of training, suddenly generalizes to the validation set. This project allows you to reproduce and study this effect.

The original Grokking experiment was conducted by [Grokking: The Surprising Truth About How the Brain Learns](https://arxiv.org/abs/2001.04451) and [Grokking: The Surprising Truth About How the Brain Learns II](https://arxiv.org/abs/2001.04452).

The model was initially trained on a generated dataset consisting of modular addition tasks ($a + b \pmod P$) where $a, b \in \mathbb{Z}_P$ and $P$ is a prime number. The dataset was generated using the `make_dataset` function in `src.data`.

### Why modular addition?

Modular addition is a simple yet effective task for studying the Grokking phenomenon because it is a non-trivial task that requires the model to learn the concept of modular arithmetic. In modular addition, the calculation can essentially be thought of as a circular addition, where the result wraps around when it exceeds the prime number $P$.

### Why prime numbers?

Prime numbers are used in modular addition to ensure that there would be no noise in our analysis. If we used composite numbers, there would be other patterns that our model would be able to exploit, which would make it easier to overfit to the training data. For example, if we used $P = 4$, the model would be able to exploit the fact that $2 + 2 = 4$ and $2 + 2 = 0$ (mod 4), which would make it easier to overfit to the training data. Using prime numbers ensures that we minimize the amount of noise in our analysis.

### Fast Fourier Transform

In our analysis, we use fast fourier transform to show how exact our W_E matrix exactly has learned the modular addition task. We plot a graph of the absolute values of the fast fourier transform of the W_E matrix, where the x-axis represents the frequency and the y-axis represents the absolute value of the fast fourier transform. We observe a few bright lines in the trained graph that are not present in the untrained graph, which indicates that the model has identified certain trigonometric patterns in the data. Which proves that the model has learned the task as intended without overfitting. Given here is the trained and untrained FFT graphs respectively.

![Alt text](graphs/trained_fft.png "Trained FFT")

![Alt text](graphs/untrained_fft.png "Untrained FFT")

### Attention Head Waves

We also plot a graph of different attention heads where the x-axis represents the attention score and the y-axis represents the `b` value in the modular addition task. We observe that the attention heads in the trained model follow a certain wave-like pattern, while the attention heads in the untrained model do not show any such pattern. It is also important to note that in this case, the `a` value, is kept constant so we can observe how the model would behave if we only changed the `b` value. This indicates that the model has learned the task as intended without overfitting. Given here are the trained and untrained attention head waves respectively.

![Alt text](graphs/trained_head_waves.png "Attention Head Waves")

![Alt text](graphs/untrained_head_waves.png "Untrained Head Waves")

## Project Structure

```
grokking_modular_addition/
├── checkpoints/          # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for analysis
│   └── analysis.ipynb    # Main analysis notebook
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── config.py         # Configuration and hyperparameters
│   ├── data.py           # Dataset generation and loading
│   ├── model.py          # Model definition (HookedTransformer)
│   └── train.py          # Training loop and utilities
├── .gitignore
├── README.md
├── requirements.txt      # Project dependencies
└── test_modularization.py # Script to verify the setup
```