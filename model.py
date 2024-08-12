from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context_len: int = 256
    # heads
    head_size: int = 64
    n_heads: int = 6
    n_layer: int = 6
    # hyperparam
    device: str = "cuda"
    dropout: float = 0.2
    learning_rate: float = 3e-4
    train_epoch: int = 10000

    def __post_init__(self):
        self.n_embd = self.n_heads * self.head_size


class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.context_len, config.context_len))
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor):
        # x: batch_size x T x n_embd
        _, T, _ = x.shape

        k: Tensor = self.key(x)  # batch_size x T x head_size
        q: Tensor = self.query(x)  # batch_size x T x head_size

        # attention scores
        wei: Tensor = (
            q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        )  # batch_size x T x T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(input=wei, dim=-1)
        wei = self.dropout(wei)  # batch_size x T x T

        v = self.value(x)  # batch_size x T x head_size

        return wei @ v  # batch_size x T x head_size


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config) for _ in range(config.n_heads)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor):
        # x: batch_size x T x n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # batch_size x T x n_embd
        return self.dropout(self.proj(out)) # batch_size x T x n_embd


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: Tensor):
        # x: batch_size x T x n_embd
        return x + self.ln2(self.ff(x + self.ln1(self.attn(x))))


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # token embedding, maps each one-hot vector to a vector of n_embd size
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # pos embedding
        self.pos_embedding = nn.Embedding(config.context_len, config.n_embd)
        # multihead attention blocks
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        # layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        # head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor = None):
        B, T = idx.shape  # batch_size x T
        # targets: batch_size, T
        tok_emb = self.token_embedding(idx)  # batch_size x T x n_embd
        pos_emb = self.pos_embedding(
            torch.arange(T, device=self.config.device)
        )  # T x n_embd

        x = tok_emb + pos_emb  # batch_size x T x n_embd

        x = self.blocks(x)  # batch_size x T x n_embd
        x = self.ln_f(x)
        logits: Tensor = self.lm_head(x)  # batch_size x context_len x vocab_size

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (batch_size x context_len) x vocab_size
            targets = targets.view(B * T)  # (batch_size x context_len)
            loss = F.cross_entropy(input=logits, target=targets)

        return logits, loss

    def generate(self, idx: Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_len :]
            # forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(input=logits, dim=1)
            # pick one based on probs
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
