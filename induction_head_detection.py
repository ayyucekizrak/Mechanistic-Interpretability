# induction_head_detection.py

# Import necessary libraries
import torch as t
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Install necessary libraries (uncomment the line below if running directly)
# !pip install transformer-lens circuitsvis matplotlib seaborn

# Set device to GPU if available
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Load a pre-trained GPT-2 model using TransformerLens
model = HookedTransformer.from_pretrained("gpt2-small").to(device)

# Sample input sequences with repeating patterns
sequences = [
    "the cat sat on the mat",
    "the cat sat the cat sat",
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps the quick brown fox jumps"
]

# Tokenize sequences
tokenized_sequences = [model.to_tokens(seq).to(device) for seq in sequences]

def dynamic_threshold_detection(activations):
    """
    Function to detect induction heads using dynamic thresholding.
    Args:
        activations: A list of attention patterns.
    Returns:
        detection_scores: Scores indicating the presence of induction heads.
        thresholds: Dynamic thresholds calculated based on attention variance.
    """
    thresholds = []
    detection_scores = []
    for act in activations:
        # Calculate attention variance and set dynamic threshold
        attention_variance = t.var(act, dim=-1).mean().item()
        threshold = attention_variance * 0.5  # Example thresholding logic
        thresholds.append(threshold)

        # Simple heuristic: High attention to previous tokens in repeating sequences indicates induction
        avg_attention = act.mean(dim=-2)  # Average over heads
        score = avg_attention.diagonal(offset=1).mean().item()  # Score based on attention to previous token
        detection_scores.append(score)
    return detection_scores, thresholds

# Extract attention activations
activations = []
for tokens in tokenized_sequences:
    _, cache = model.run_with_cache(tokens)
    # Access the attention patterns for the last layer and all heads
    attn_pattern = cache["blocks.11.attn.hook_pattern"]
    activations.append(attn_pattern)

# Detect induction heads using dynamic thresholding
predicted_scores, thresholds = dynamic_threshold_detection(activations)
predicted_labels = [1 if score > threshold else 0 for score, threshold in zip(predicted_scores, thresholds)]

# Dummy true labels for demonstration (1 = induction head, 0 = not)
true_labels = [0, 1, 0, 1]

# Calculate evaluation metrics
precision = precision_score(true_labels, predicted_labels, zero_division=1)
recall = recall_score(true_labels, predicted_labels, zero_division=1)
f1 = f1_score(true_labels, predicted_labels, zero_division=1)

print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Visualize attention patterns using matplotlib
for i, pattern in enumerate(activations):
    # Reduce the dimensions to 2D by selecting one head or averaging across heads
    attention_matrix = pattern.mean(dim=1)[0].cpu().detach().numpy()  # Select the first head after averaging across heads
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, annot=False, cmap='Blues', 
                xticklabels=model.to_str_tokens(tokenized_sequences[i]), 
                yticklabels=model.to_str_tokens(tokenized_sequences[i]))
    plt.title(f"Activation Pattern for Sequence {i+1}")
    plt.show()
