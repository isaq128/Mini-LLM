import torch
from config import *
from model import MiniGPT
from utils import *
import os

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars, stoi, itos = build_vocab(text)
vocab_size = len(chars)

data = torch.tensor(encode(text, stoi), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        loss_list = []
        for _ in range(200):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            loss_list.append(loss.item())
        losses[split] = sum(loss_list) / len(loss_list)
    model.train()
    return losses

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train {losses['train']:.4f}, Val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos
}, "checkpoints/mini_llm.pt")

print("Model saved.")