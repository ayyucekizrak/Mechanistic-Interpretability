#### This project, developed by [Merve Ayyüce Kızrak](https://www.linkedin.com/in/merve-ayyuce-kizrak/) as part of the [AI Alignment Course - AI Safety Fundamentals powered by BlueDot Impact](https://aisafetyfundamentals.com/), leverages a range of advanced resources to explore key concepts in mechanistic interpretability in transformers.

**To access more detailed information and comments on the analysis results, read the project's blog post: [Mechanistic Interpretability in Action: Understanding Induction Heads and QK Circuits in Transformers](https://medium.com/)**

### Acknowledgment

I would like to express my gratitude to the AI Safety Fundamental team, the facilitators in the cohorts I participated in, and all the participants for their contributions to developing new ideas in our discussions. I am pleased to be a part of this team.

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

<img align="middle" src="https://cdn-images-1.medium.com/v2/resize:fit:800/0*4trUuXdwBd43DHEs.png"> 

*The way an induction head in transformer models pays attention to repeated patterns in a sequence is presented in the image. When a sequence of tokens is repeated, the induction head notices the repetition. Thus, it shifts its attention to the corresponding token in the previous sequence, and the probability of predicting the next token based on the previously attended pattern, i.e., logit, increases. This mechanism helps the model to remember and repeat sequences during in-context learning.*

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

<img align="middle" src="https://cdn-images-1.medium.com/v2/resize:fit:800/0*wi0zyL1u0oSDHQ3j.png"> 

*It shows how QK (Query-Key) circuits in transformer models attend to different tokens according to their relevance. The attention pattern is visualized as moving information from the "key" token to the "query" token, affecting the model's prediction for the next token, the logit effect. This mechanism shows how attention is directed to specific words, thus affecting how the model processes and predicts the following tokens in a sequence.*

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

### Reference
1. [Induction Heads - Illustrated by Callum McDougall](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated)
2. [Zoom-In: An Introduction to Circuits - OpenAI](https://distill.pub/2020/circuits/zoom-in/)
3. [Intro to Mechanistic Interpretability: TransformerLens & induction circuits](https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch)
4. [TransformerLens Library by Neel Nanda](https://github.com/TransformerLensOrg/TransformerLens)
5. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
6. [OpenAI's Research on Transformer Interpretability](https://openai.com/research/)
7. [Anthropic's Research on Transformer Behaviors](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
8. [EleutherAI Research on Large Language Models](https://www.eleuther.ai/)
9. [QK Circuit Analysis for Attention Allocation in Transformers](https://arxiv.org/abs/1706.03762)
10. [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
11. [DeepMind's Research on Model Interpretability](https://deepmind.google/research/publications/22295/)
12. [Distill Articles on Neural Network Interpretability](https://distill.pub/2020/circuits/)
