import torch

from model import ModelConfig, GPT

torch.manual_seed(42)

EVAL_NUM_BATCH = 500

with open("input.txt", "r", encoding="utf-8") as f:
    data = f.read()

chars = sorted(list(set(data)))
VOCAB_SIZE = len(chars)

stoi = { ch: i for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda i: "".join([chars[x] for x in i])
data = torch.tensor(encode(data), dtype=torch.long)

cutoff = int(len(data) * 0.9)
TRAIN_DATA, VAL_DATA = data[:cutoff], data[cutoff:]


def get_batch(split: str, config: ModelConfig):
    data = TRAIN_DATA if split == "train" else VAL_DATA
    idx = torch.randint(len(data)-config.context_len, (config.context_len,))
    x = torch.stack([data[i:i+config.context_len] for i in idx])
    y = torch.stack([data[i+1:i+config.context_len+1] for i in idx])
    return x.to(config.device), y.to(config.device)


@torch.no_grad()
def estimate_loss(model: GPT):
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(EVAL_NUM_BATCH)
        for k in range(EVAL_NUM_BATCH):
            X, Y = get_batch(split)
            loss: torch.Tensor
            _, loss = model(X, Y)
            losses[k] = loss.item()
        avg_loss = losses.mean()
        print(f"{split} loss: {avg_loss:.4f}")
    model.train()


def train(save: bool = True):
    config = ModelConfig(vocab_size=VOCAB_SIZE)
    model = GPT(config=config)
    m = model.to(config.device)

    print(
        "This model has ", sum(p.numel() for p in m.parameters()) / 1e6, "M parameters"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.train_epoch):
        if iter % EVAL_NUM_BATCH == 0 or iter == config.train_epoch - 1:
            estimate_loss(model=m)
        
        xb, yb = get_batch("train")
        
        loss: torch.Tensor
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        optimizer.step()
    
    if save:
        torch.save(m, 'model.pth')
        
    return m