from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context_len: int = 256
    n_embd: int = 384
    n_head: int = 6
    head_size: int = 384
    block_size: int = 10
    n_layer: int = 6
    # hyperparam
    device: str = "cuda"
    dropout: float = 0.2


class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor):
        B, T, C = x.shape


class Block(nn.Module):
    pass


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

        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.config.device))

        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

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
