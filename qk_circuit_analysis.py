# qk_circuit_analysis.py

# Import necessary libraries
import torch as t
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
import matplotlib.pyplot as plt
import seaborn as sns

# Install necessary libraries (uncomment the line below if running directly)
# !pip install transformer-lens circuitsvis matplotlib seaborn

# Set device to GPU if available
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Load a pre-trained GPT-2 model using TransformerLens
model = HookedTransformer.from_pretrained("gpt2-small").to(device)

# Sample input sequences
sequences = [
    "the cat sat on the mat",
    "the cat sat the cat sat",
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps the quick brown fox jumps"
]

# Tokenize sequences
tokenized_sequences = [model.to_tokens(seq).to(device) for seq in sequences]

def extract_qk_patterns(model, tokens, layer):
    """
    Function to extract Query and Key matrices for a given layer.
    Args:
        model: Transformer model.
        tokens: Tokenized input sequence.
        layer: Target layer number.
    Returns:
        Q: Query matrix.
        K: Key matrix.
    """
    # Get Query and Key matrices from the cache
    _, cache = model.run_with_cache(tokens)
    Q = cache[f'blocks.{layer}.attn.hook_q']
    K = cache[f'blocks.{layer}.attn.hook_k']
    return Q, K

# Example: Extract QK patterns from layer 11
layer = 11
QK_patterns = [extract_qk_patterns(model, tokens, layer) for tokens in tokenized_sequences]

def plot_qk_interactions(Q, K, tokens, title="QK Interaction Heatmap"):
    """
    Function to visualize QK interactions as heatmaps.
    Args:
        Q: Query matrix.
        K: Key matrix.
        tokens: Tokenized sequence as strings.
        title: Title for the heatmap.
    """
    # Compute QK dot products for each head and then average
    QK_dot = t.einsum('bhqd,bhkd->bhqk', Q, K).mean(dim=0).mean(dim=0).cpu().numpy()

    # Visualize as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(QK_dot, annot=False, cmap='Blues', xticklabels=tokens, yticklabels=tokens)
    plt.title(title)
    plt.show()

# Plot interactions for each sequence
for i, (Q, K) in enumerate(QK_patterns):
    tokens_str = model.to_str_tokens(tokenized_sequences[i])
    plot_qk_interactions(Q, K, tokens_str, title=f"QK Interaction for Sequence {i+1}")
