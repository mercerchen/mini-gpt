# Simple tests. No need for pytest
import torch

from model import GPT
from train import encode, decode

if __name__ == "__main__":
    # assert decode(encode("hello")) == "hello"

    model: GPT = torch.load(f='model.pth')

    model.eval()
    text = "诸葛村夫"
    completions = model.generate(torch.tensor([encode(text)], dtype=torch.long, device='cuda'), max_new_tokens=500)
    print(completions)
    print(decode(completions[0].tolist()))
