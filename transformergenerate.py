import json
import torch
from transformermodels import PasswordTransformer


weights_path = "transformer_weights.pth"
vocab_path   = "transformer_vocab.json"
output_file  = "transformer_generated_passwords.txt"
num_generate = 100_000
# how many passwords to sample at once (lower if OOM)
batch_size   = 1024 
# > 1 = more random, < 1 = more conservative  
temperature  = 1.0 
seq_len      = 10

# 1. Load vocabulary
with open(vocab_path, "r", encoding="utf-8") as f:
    idx2char_list = json.load(f)
idx2char   = {idx: ch for idx, ch in enumerate(idx2char_list)}
vocab_size = len(idx2char_list)
# index 0 is always 'unk' / padding
unk_idx    = 0  

# 2. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating on: {device}")

model = PasswordTransformer(vocab_size, seq_len).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print(f"Loaded model from {weights_path}")

# 3. Generate in batches
all_passwords = []
generated_so_far = 0

while generated_so_far < num_generate:
    this_batch = min(batch_size, num_generate - generated_so_far)
    token_ids  = model.generate(this_batch, device, temperature=temperature, unk_idx=unk_idx)

    for row in token_ids:
        # convert indices → chars, strip trailing padding (unk)
        pwd = "".join(idx2char[idx.item()] for idx in row)
        pwd = pwd.replace(idx2char[unk_idx], "")   # strip padding chars
        all_passwords.append(pwd)

    generated_so_far += this_batch
    if generated_so_far % 10_000 == 0:
        print(f"  Generated {generated_so_far}/{num_generate}...")

# 4. Save
with open(output_file, "w", encoding="utf-8") as f:
    for p in all_passwords:
        f.write(p + "\n")

print(f"\nDone. {num_generate} passwords saved to {output_file}")
print(f"Sample:\n  " + "\n  ".join(all_passwords[:10]))
