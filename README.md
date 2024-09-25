#### This project, developed by [Merve Ayyüce Kızrak](https://www.linkedin.com/in/merve-ayyuce-kizrak/) as part of the [AI Alignment Course - AI Safety Fundamentals powered by BlueDot Impact](https://aisafetyfundamentals.com/), leverages a range of advanced resources to explore key concepts in mechanistic interpretability in transformers.

**Here is the blog post of the project: [Mechanistic Interpretability in Action: Understanding Induction Heads and QK Circuits in Transformers](https://medium.com/)**

---

# Mechanistic Interpretability in Action: Understanding Induction Heads and QK Circuits in Transformers

## Overview
This repository contains two projects aimed at enhancing the mechanistic interpretability of transformer-based models, specifically focusing on GPT-2. The projects provide insights into two critical aspects of transformer behavior: **Induction Head Detection** and **QK Circuit Analysis**. By understanding these mechanisms, we aim to make transformer models more transparent, interpretable, and aligned with human values.

## Step 1 - Induction Head Detection
Induction heads are specialized attention heads within transformer models that help maintain and repeat sequences during in-context learning. This project focuses on identifying these heads and visualizing their behavior in repetitive sequences.

Here is the notebook! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ayyucekizrak/Mechanistic-Interpretability/blob/main/induction_head_detection.ipynb)

### Key Features
- **Dynamic Threshold Detection:** A novel method to detect induction heads using dynamic thresholds based on attention score variance.
- **Attention Heatmaps:** Visualizations of attention patterns to highlight induction head activity in repetitive sequences.
- **Evaluation Metrics:** Precision, recall, and F1-score calculations to measure the effectiveness of the detection method.

### How to Run
1. **Install Required Libraries:**
```bash
   pip install transformer-lens circuitsvis matplotlib seaborn
```
3. **Execute the Script:**
Run the `qk_circuit_analysis.py` script to extract and visualize QK interactions for the provided sample sequences.

4. **Modify Input Sequences:**
Change the sequences variable in the script to analyze your own text sequences for QK circuit interactions.

### Visualization
The script generates attention heatmaps showing induction head activity for each sequence, providing a visual representation of how the model captures and retains context.

## Step 2 -  QK Circuit Analysis
QK (Query-Key) circuits are fundamental to how transformers allocate attention among tokens in an input sequence. This project focuses on analyzing and visualizing QK interactions to understand how transformers prioritize information.

Here is the notebook! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/ayyucekizrak/Mechanistic-Interpretability/blob/main/qk_circuit_analysis.ipynb) 

### Key Features
- **QK Pattern Extraction:** Extracts Query and Key matrices from a specified layer of the transformer model.
- **QK Interaction Heatmaps:** Visualizations of the interaction between Query and Key matrices to showcase attention distribution.
- **Causal Interventions:** Analyzes the impact of QK circuits on model behavior through causal interventions like ablations.

### How to Run
1. **Install Required Libraries:**
```bash
pip install transformer-lens circuitsvis matplotlib seaborn
```
2. **Execute the Script:** Run the `qk_circuit_analysis.py` script to extract and visualize QK interactions for the provided sample sequences.
3. **Modify Input Sequences:** Change the sequences variable in the script to analyze your own text sequences for QK circuit interactions.

### Visualization
The script generates QK interaction heatmaps for each sequence, highlighting how attention is distributed among tokens based on the model's Query and Key matrices.

### Use Cases
- **Text Generation:** Improve understanding of how models retain and repeat context over long sequences.
- **Machine Translation:** Analyze how models allocate attention to relevant parts of a sentence.
- **Sentiment Analysis:** Understand how specific tokens are prioritized based on context.

### Installation
Clone the repository and install the necessary dependencies:
```bash
git clone <repository-link>
pip install -r requirements.txt
```

### Usage
1. Choose a project directory (`induction_heads` or `qk_circuits`).
2. Run the respective script for the project you are interested in:
- `induction_head_detection.py` for Induction Heads.
- `qk_circuit_analysis.py` for QK Circuits.
3. Modify the sequences variable in the script to use your own text sequences.

### Future Work
- **Larger Transformer Models:** Extend the analysis to larger models like GPT-3 or T5.
- **Fine-Tuned Causal Interventions:** Implement more precise causal interventions to isolate the effects of specific model components.
- **Bias and Fairness Analysis:** Explore the impact of induction heads and QK circuits on model bias and fairness.
- **Cross-Modal Analysis:** Apply these techniques to cross-modal transformers that handle both vision and language tasks.

###  Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

###  Contact
For any questions or inquiries, please contact [ayucekizrak@gmail.com].
