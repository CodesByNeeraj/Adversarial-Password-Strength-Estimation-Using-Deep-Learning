import random

INPUT_FILE  = "password_exp/hashcat_generated.txt"
OUTPUT_FILE = "password_exp/hashcat_generated_sample.txt"
SAMPLE_SIZE = 2_381_844
SEED        = 42

print(f"Loading {INPUT_FILE} ...")
valid = []
skipped = 0

with open(INPUT_FILE, "rb") as f:
    for line in f:
        try:
            pwd = line.decode("utf-8").strip()
            if pwd:
                valid.append(pwd)
        except UnicodeDecodeError:
            skipped += 1

print(f"Valid UTF-8 passwords: {len(valid):,}")
print(f"Skipped (non-UTF-8):   {skipped:,}")

if SAMPLE_SIZE > len(valid):
    raise ValueError(f"Requested {SAMPLE_SIZE:,} but only {len(valid):,} valid passwords available.")

sampled = random.Random(SEED).sample(valid, SAMPLE_SIZE)
print(f"Sampled:               {len(sampled):,}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(sampled) + "\n")

print(f"Saved to {OUTPUT_FILE}")
