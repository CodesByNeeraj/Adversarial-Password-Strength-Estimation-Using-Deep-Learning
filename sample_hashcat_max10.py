import random

INPUT_FILE  = "password_exp/hashcat_15M_sample_max10.txt"
OUTPUT_FILE = "password_exp/hashcat_1e7_sample.txt"
SAMPLE_SIZE = 9153003
SEED        = 42

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    passwords = [line.strip() for line in f if line.strip()]

print(f"Total loaded: {len(passwords):,}")

if SAMPLE_SIZE > len(passwords):
    raise ValueError(f"Requested {SAMPLE_SIZE:,} but only {len(passwords):,} available.")

sampled = random.Random(SEED).sample(passwords, SAMPLE_SIZE)
print(f"Sampled:      {len(sampled):,}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(sampled) + "\n")

print(f"Saved to {OUTPUT_FILE}")
