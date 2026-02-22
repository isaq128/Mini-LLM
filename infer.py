import torch
from config import *
from model import MiniGPT
from utils import *

checkpoint = torch.load("checkpoints/mini_llm.pt", map_location=device)
stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

vocab_size = len(stoi)
model = MiniGPT(vocab_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

while True:
    prompt = input("\nPrompt: ")
    context = torch.tensor([encode(prompt, stoi)], dtype=torch.long).to(device)
    generated = model.generate(context, max_new_tokens=300)
    print("\nResponse:\n")
    print(decode(generated[0].tolist(), itos))