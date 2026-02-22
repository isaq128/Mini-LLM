import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
block_size = 64
batch_size = 32

# Model
n_embd = 128
n_head = 4
n_layer = 2
dropout = 0.2

# Training
learning_rate = 3e-4
max_iters = 4000
eval_interval = 500

# Generation
temperature = 1.0
top_k = 20