MiniLLM – Transformer Language Model From Scratch

MiniLLM is a character-level GPT-style Transformer implemented entirely in PyTorch without using pre-built transformer libraries.

🔥 Features

Token + positional embeddings

Multi-head self-attention (implemented manually)

Feedforward layers

Residual connections & LayerNorm

Autoregressive generation

Temperature + Top-k sampling

Training + validation loop

Checkpoint saving & loading

CLI-based inference

🏗 Architecture

2 Transformer Blocks

4 Attention Heads

Embedding Dimension: 128

Context Window: 64 tokens

Dropout: 0.2

📉 Training

Trained on Shakespeare corpus.

Final Validation Loss: ~1.9

▶️ Run Training
python train.py
💬 Run Inference
python infer.py
