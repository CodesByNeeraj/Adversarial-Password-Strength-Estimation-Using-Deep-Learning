import random

INPUT_FILE  = "data/gpt2_generated_1e6.txt"
OUTPUT_FILE = "data/gpt2_generated_928660.txt"
SAMPLE_SIZE = 928_660
SEED        = 42

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    passwords = list(set(line.strip() for line in f if line.strip()))

print(f"Unique loaded: {len(passwords):,}")

if SAMPLE_SIZE > len(passwords):
    raise ValueError(f"Requested {SAMPLE_SIZE:,} but only {len(passwords):,} unique passwords available.")

sampled = random.Random(SEED).sample(passwords, SAMPLE_SIZE)
print(f"Sampled:       {len(sampled):,}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(sampled) + "\n")

print(f"Saved to {OUTPUT_FILE}")
