🚀 MiniLLM
Transformer Language Model Built From Scratch (PyTorch)
🧠 Overview
MiniLLM is a character-level GPT-style Transformer
implemented entirely from scratch using PyTorch.

No Hugging Face.
No pre-built Transformer modules.
All attention mechanisms implemented manually.

MiniLLM learns to model:

P(next_token | previous_tokens)

It predicts the next character using a decoder-only Transformer architecture.

🏗 Architecture
Input Tokens
     ↓
Token Embedding
     ↓
Positional Embedding
     ↓
Transformer Block × 2
     ├── Multi-Head Self-Attention
     ├── Feedforward Network
     ├── Residual Connections
     └── LayerNorm
     ↓
Linear Output Layer
     ↓
Softmax
🔬 Model Configuration
Embedding Dimension  : 128
Transformer Blocks   : 2
Attention Heads      : 4
Context Window       : 64 tokens
Dropout              : 0.2
Optimizer            : AdamW
Learning Rate        : 3e-4
📉 Training
Dataset        : Shakespeare Corpus
Training Steps : 4000
Batch Size     : 32
Final Val Loss : ~1.9

Model shows stable convergence and meaningful structure learning.

🔁 Generation Strategy
1. Feed prompt
2. Compute logits
3. Apply temperature scaling
4. Apply Top-k filtering
5. Sample next token
6. Repeat

Supports:

✔ Temperature control
✔ Top-k sampling
✔ Autoregressive decoding

▶️ Run

Install

pip install torch

Train

python train.py

Infer

python infer.py
🎯 What This Demonstrates
✔ Transformer implementation from scratch
✔ Multi-head self-attention mechanics
✔ Residual + LayerNorm structure
✔ Autoregressive language modeling
✔ Practical training pipeline
🚀 Future Improvements
• BPE Tokenization
• Larger model scaling
• Web deployment
• Domain fine-tuning
• Perplexity tracking
