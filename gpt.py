# Creating a GPT from scratch

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# We are training our gpt on a sample Shakespeare text
# LOAD THE DATA
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# IMPLEMENTING A TOKENIZER
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# input string, output list of integers
def encode(s): return [stoi[c] for c in s]
# input list of integers, output string
def decode(ls): return [''.join(itos[i] for i in ls)]


# Encode the dataset and store it in torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Separate data into train and validation sets
n = int(0.8 * len(data))  # use 80% for training and rest for testing
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    # Get data from the relevant set
    data = train_data if split == "train" else val_data
    # ix => random set of data of batch size lying in a block
    ix = torch.randint(len(data)-block_size, (batch_size,))
    # Put x, y as rows of a 4x8 tensor using torch.stack()
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)  # move the data to 'gpu' or 'cpu'
    return x, y

# Context manager, torch.no_grad() tells pytorch to not call .backward()
# on this block of code


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# IMPLEMENT A BIGRAM LANGUAGE MODEL
class BigramLanguageModel(nn.Module):
    # contructor
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads logits(scores) for next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # arrange the logits in a Batch(B) x Time(T) x Channel(C) tensor
        # B = 4, T = 8, C = vocab_size(65 in our case)
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # Reshaping logits and targets for cross entropy function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Now we also need to measure loss using cross entropy function
            # This measures negative log likelihood
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Generate data from the model
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus on last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sample index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel(vocab_size)
model.to(device)  # move the model parameters to gpu

# TRAINING THE MODEL
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {
              losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        # zero out all the gradients from previous step
        optimizer.zero_grad(set_to_none=True)
        # get gradients for all the parameters
        loss.backward()
        # use the gradients to update parameters
        optimizer.step()


# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
