INPUT_FILE  = "data/2020LeakedPw_10M_filtered.txt"
OUTPUT_FILE = "data/2020LeakedPw_10M_filtered_max10.txt"
MAX_LEN     = 10

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    passwords = [line.strip() for line in f if line.strip()]

print(f"Total loaded:   {len(passwords):,}")

filtered = [p for p in passwords if len(p) <= MAX_LEN]

print(f"Kept (<=10):    {len(filtered):,}")
print(f"Removed (>10):  {len(passwords) - len(filtered):,}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(filtered) + "\n")

print(f"Saved to {OUTPUT_FILE}")
