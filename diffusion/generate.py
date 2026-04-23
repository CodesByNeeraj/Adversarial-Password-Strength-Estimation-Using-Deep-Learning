import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import PasswordVocab
from model import DiffusionTransformer

#config(must match train.py)
VOCAB_FILE   = "diffusion/vocab.json"
WEIGHTS_FILE = "diffusion/model_weights.pth"
OUTPUT_FILE  = "diffusion/generated_passwords_1e6.txt"

SEQ_LEN      = 10
HIDDEN_DIM   = 256
N_HEADS      = 8
N_LAYERS     = 6
T            = 1000
BETA_START   = 1e-4
BETA_END     = 0.02

NUM_GENERATE = 1000000
BATCH_SIZE   = 1024
#evenly-spaced timesteps from T-1 → 0
SAMPLE_STEPS = 200 


#noise schedule (same as train.py)
def make_alpha_bar(T, beta_start, beta_end, device):
    betas     = torch.linspace(beta_start, beta_end, T, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


#single reverse step: x_t → x_{t_next}
@torch.no_grad()
def reverse_step(x_t, t_curr, t_next, model, vocab, alpha_bar, device, batch_size):
    B = x_t.size(0)
    t_tensor = torch.full((B,), t_curr, dtype=torch.long, device=device)
    #(B, seq_len, vocab_size)
    logits = model(x_t, t_tensor)             

    #mask out [PAD] and [MASK] from predictions so model never outputs them
    logits[:, :, vocab.pad_id]  = -1e9
    logits[:, :, vocab.mask_id] = -1e9

    #(B, seq_len, vocab_size)
    probs   = F.softmax(logits, dim=-1)      
    #sample x_0 prediction for every position
    x0_pred = torch.multinomial(
        probs.view(-1, vocab.vocab_size), num_samples=1
    ).view(B, SEQ_LEN)                        # (B, seq_len)

    if t_next < 0:
        #final step: reveal everything
        return x0_pred
    #scalar ᾱ_t
    alpha_curr = alpha_bar[t_curr]    
    #scalar ᾱ_{t_next}     
    alpha_next = alpha_bar[t_next]         
    #(B, seq_len) bool
    masked_positions = (x_t == vocab.mask_id)

    #probability of unmasking at this step (from absorbing diffusion posterior)
    #p(unmask) = (ᾱ_{t_next} - ᾱ_t) / (1 - ᾱ_t)
    denom      = (1.0 - alpha_curr).clamp(min=1e-9)
    p_unmask   = ((alpha_next - alpha_curr) / denom).clamp(0.0, 1.0)

    #bernoulli draw: unmask this position?
    unmask = torch.bernoulli(
        torch.full_like(x_t, p_unmask.item(), dtype=torch.float)
    ).bool()

    #only unmask positions that are currently masked
    do_unmask = masked_positions & unmask

    x_next = x_t.clone()
    #reveal predicted token
    x_next[do_unmask] = x0_pred[do_unmask] 
    #still-masked positions stay as [MASK]
    return x_next


#full generation
@torch.no_grad()
def generate(model, vocab, alpha_bar, device):
    model.eval()

    #evenly spaced timesteps T-1 → 0 (descending)
    step_size = max(T // SAMPLE_STEPS, 1)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        with tqdm(total=NUM_GENERATE, desc="Generating") as pbar:
            generated = 0
            while generated < NUM_GENERATE:
                B = min(BATCH_SIZE, NUM_GENERATE - generated)

                #Start: fully masked sequence
                x = torch.full(
                    (B, SEQ_LEN), vocab.mask_id,
                    dtype=torch.long, device=device
                )

                #Iterative denoising
                for i, t_curr in enumerate(timesteps):
                    t_next = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                    x = reverse_step(x, t_curr, t_next, model, vocab,
                                     alpha_bar, device, B)

                #Decode and write
                for row in x:
                    pwd = vocab.decode(row.tolist())
                    if pwd:
                        out_f.write(pwd + '\n')
                        generated += 1
                        pbar.update(1)
                        if generated >= NUM_GENERATE:
                            break

    print(f"Done. {generated:,} passwords saved to {OUTPUT_FILE}")


#Entry point
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating on: {device}")

    vocab = PasswordVocab()
    vocab.load(VOCAB_FILE)
    print(f"Vocab size: {vocab.vocab_size}")

    model = DiffusionTransformer(
        vocab.vocab_size, HIDDEN_DIM, N_HEADS, N_LAYERS, SEQ_LEN
    ).to(device)
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    print("Model loaded.")

    alpha_bar = make_alpha_bar(T, BETA_START, BETA_END, device)
    generate(model, vocab, alpha_bar, device)
