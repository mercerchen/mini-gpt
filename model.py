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
    n_head: int = 6
    n_layer: int = 6
    # hyperparam
    device: str = "cuda"
    dropout: float = 0.2
    learning_rate: float = 3e-4
    train_epoch: int = 10000

    def __post_init__(self):
        self.n_embd = self.n_head * self.head_size


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
        _, T, C = x.shape
        assert T == self.config.context_len and C == self.config.n_embd, (
            f"Input is of T: {T}, C: {C}"
            f" but should be of T: {self.config.context_len} and C: {self.config.n_embd}"
        )

        k: Tensor = self.key(x)  # B, T, head_size
        q: Tensor = self.query(x)

        # attention scores
        wei: Tensor = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(input=wei, dim=-1)
        wei = self.dropout(wei)  # B, T, T

        v = self.value(x)  # (B, T, T) x (B, T, n_embd) -> (B, T, n_embd)

        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config.head_size) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


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
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: Tensor):
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
            *[Block(config.n_embd, n_head=config.n_head) for _ in range(config.n_layer)]
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
        B, T = idx.shape
        # targets: B, T
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.config.device))

        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits: Tensor = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
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
