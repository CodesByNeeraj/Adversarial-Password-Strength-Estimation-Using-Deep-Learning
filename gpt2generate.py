import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_PATH     = "gpt2_password_model"
OUTPUT_FILE    = "data/gpt2_generated_1e4.txt"
#adjustable number. for trend analysis
NUM_GENERATE   = 10000
BATCH_SIZE     = 256
#10 chars + 1 EOS (each token = 1 char after char-level retrain)
MAX_NEW_TOKENS = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Generating on: {device}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("Model loaded.")

#seed characters - must match what the model saw at training time.
#gpt2train.py trains WITHOUT a BOS prefix, so GPT-2's bos_token (50256,
#which equals eos_token) is completely out-of-distribution and causes the
#full-dataset model to fall back to web-text subword patterns like "ing",
#"ed", "to" instead of real passwords.
#fix:we seed each generation with a random password-like first character so
#the model conditions on an in-distribution starting token.

SEED_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "!@#$_."
)

#pre-tokenize all seed chars (each is a single token in GPT-2 BPE)
seed_token_ids = []
for ch in SEED_CHARS:
    ids = tokenizer.encode(ch, add_special_tokens=False)
    if ids:
        seed_token_ids.append(ids[0])

generated_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    with tqdm(total=NUM_GENERATE, desc="Generating") as pbar:
        while generated_count < NUM_GENERATE:
            batch = min(BATCH_SIZE, NUM_GENERATE - generated_count)

            #sample a random in-distribution first character per item
            seeds = random.choices(seed_token_ids, k=batch)
            input_ids = torch.tensor(
                [[tok] for tok in seeds], dtype=torch.long
            ).to(device)
            attention_mask = torch.ones((batch, 1), dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            for output in outputs:
                text = tokenizer.decode(output, skip_special_tokens=True)
                #strip spaces: char-level training uses single-byte tokens but GPT-2
                #may still decode some with a leading Ġ (space prefix). Removing
                #spaces reconstructs the original password character sequence.
                pwd = text.replace(" ", "").split("\n")[0][:10]
                if pwd:
                    f.write(pwd + "\n")
                    generated_count += 1
                    pbar.update(1)

print(f"Done. Saved {generated_count:,} passwords to {OUTPUT_FILE}")
