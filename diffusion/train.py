import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import PasswordVocab, PasswordDataset
from model import DiffusionTransformer

#config
DATA_FILE    = "data/train.txt"
VOCAB_FILE   = "diffusion/vocab.json"
WEIGHTS_FILE = "diffusion/model_weights.pth"

SEQ_LEN    = 10
HIDDEN_DIM = 256
N_HEADS    = 8
N_LAYERS   = 6
BATCH_SIZE = 512
LR         = 1e-4
EPOCHS     = 10
#total diffusion timesteps
T          = 1000 
BETA_START = 1e-4
BETA_END   = 0.02


#Noise schedule
def make_alpha_bar(T, beta_start, beta_end, device):
    betas     = torch.linspace(beta_start, beta_end, T, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)  # ᾱ_t: prob of staying unmasked
    return alpha_bar



#forward diffusion q(x_t | x_0)
def q_sample(x0, t, alpha_bar, mask_id):

    #(B,)
    alpha_t  = alpha_bar[t]    
    #(B, seq_len) bool                    
    keep     = torch.bernoulli(                          
        alpha_t.unsqueeze(1).expand_as(x0).float()
    ).bool()
    mask_tok = torch.full_like(x0, mask_id)
    return torch.where(keep, x0, mask_tok)

#training loop
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    #build vocabulary from training data
    print(f"Loading passwords from {DATA_FILE} ...")
    with open(DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        passwords = [
            line.strip() for line in f
            if line.strip()
            and len(line.strip()) <= SEQ_LEN
            and line.strip().isascii()
        ]
    print(f"Loaded {len(passwords):,} passwords")

    vocab = PasswordVocab()
    vocab.build(passwords)
    print(f"Vocab size: {vocab.vocab_size}")
    vocab.save(VOCAB_FILE)
    print(f"Vocab saved to {VOCAB_FILE}")

    dataset = PasswordDataset(DATA_FILE, vocab)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=device.type == 'cuda',
    )

    model = DiffusionTransformer(
        vocab.vocab_size, HIDDEN_DIM, N_HEADS, N_LAYERS, SEQ_LEN
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    alpha_bar = make_alpha_bar(T, BETA_START, BETA_END, device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss  = 0.0
        total_batches = 0

        for batch in loader:
            #(B, seq_len)
            x0 = batch.to(device)          
            B  = x0.size(0)

            t   = torch.randint(0, T, (B,), device=device)
            x_t = q_sample(x0, t, alpha_bar, vocab.mask_id)
            #(B, seq_len, vocab_size)
            logits = model(x_t, t)       

            #loss only on positions that were masked
            masked = (x_t == vocab.mask_id)
            if masked.sum() == 0:
                continue

            loss = F.cross_entropy(
                #(N_masked, vocab_size)
                logits[masked],     
                #(N_masked,)      
                x0[masked],               
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(total_batches, 1)
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), WEIGHTS_FILE)
    print(f"Weights saved to {WEIGHTS_FILE}")


if __name__ == '__main__':
    train()
