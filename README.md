🚀 MiniLLM – Transformer Language Model Built From Scratch

MiniLLM is a character-level Transformer-based Language Model implemented entirely from scratch using PyTorch.
No Hugging Face. No pre-built Transformer blocks. All core components are manually implemented to deeply understand modern LLM architecture.

This project demonstrates the internal mechanics of GPT-style models, including multi-head self-attention, positional embeddings, and autoregressive generation.

🧠 Project Overview

MiniLLM learns to model:

It predicts the next character given previous context using a Transformer architecture.

The model was trained on a Shakespeare corpus and generates structured dialogue-style text.

🏗 Architecture

MiniLLM follows a GPT-style decoder-only Transformer architecture.

🔹 Core Components

Token Embedding Layer

Positional Embedding Layer

Multi-Head Self-Attention

Feedforward Neural Network

Residual Connections

Layer Normalization

Autoregressive Output Head

🔬 Model Configuration
Component	Value
Embedding Dimension	128
Number of Transformer Blocks	2
Attention Heads	4
Context Window (Block Size)	64
Dropout	0.2
Optimizer	AdamW
Learning Rate	3e-4
🔍 Attention Mechanism

Each Transformer block includes multi-head self-attention:

Queries, Keys, Values computed via linear projections

Scaled dot-product attention

Causal masking to prevent future token leakage

Softmax normalization

Weighted value aggregation

Mathematically:

𝐴
𝑡
𝑡
𝑒
𝑛
𝑡
𝑖
𝑜
𝑛
(
𝑄
,
𝐾
,
𝑉
)
=
𝑆
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=Softmax(
d
k
	​

	​

QK
T
	​

)V
📉 Training Details

Dataset: Shakespeare corpus

Training Iterations: 4000

Batch Size: 32

Validation Strategy: Periodic loss estimation

Final Validation Loss:

~1.9

This demonstrates stable convergence and meaningful pattern learning.

🔁 Text Generation

MiniLLM generates text autoregressively:

Feed input prompt

Compute next-token probability distribution

Apply temperature scaling

Apply Top-k filtering

Sample next token

Repeat

Supports:

Temperature control

Top-k sampling

▶️ How To Run
Install Dependencies
pip install torch
Train Model
python train.py
Run Inference
python infer.py
🎯 Key Learnings

Deep understanding of self-attention mechanics

Transformer block construction

Autoregressive language modeling

Decoding strategies (Temperature, Top-k)

Training stability & loss behavior

Modular model structuring

🚀 Future Improvements

Byte-Pair Encoding (BPE) tokenization

Larger context window

More Transformer layers

Perplexity tracking

Web-based interface (FastAPI / Streamlit)

Domain-specific fine-tuning

📌 Why This Project Matters

Many LLM projects rely on high-level libraries.

MiniLLM focuses on understanding the internal mechanics of Transformer-based language models by implementing all critical components manually.

This project demonstrates strong foundational knowledge of:

Deep Learning

Sequence Modeling

Transformer Architectures

PyTorch Internals

Generative AI Systems

