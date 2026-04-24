"""
filter_2020leak.py

Removes from data/2020LeakedPw_10M.txt any password that already appears
in the RockYou dataset (data/fullData.txt), then writes the deduplicated
remainder to data/2020LeakedPw_10M_filtered.txt.
"""

ROCKYOU_FILE  = "data/fullData.txt"
LEAK_FILE     = "data/2020LeakedPw_10M.txt"
OUTPUT_FILE   = "data/2020LeakedPw_10M_filtered.txt"

print(f"Loading RockYou from {ROCKYOU_FILE} ...")
with open(ROCKYOU_FILE, "r", encoding="utf-8", errors="ignore") as f:
    rockyou_set = set(line.strip() for line in f if line.strip())
print(f"  RockYou unique passwords: {len(rockyou_set):,}")

print(f"\nLoading 2020 leak from {LEAK_FILE} ...")
with open(LEAK_FILE, "r", encoding="utf-8", errors="ignore") as f:
    leak_passwords = [line.strip() for line in f if line.strip()]
print(f"  2020 leak total lines:    {len(leak_passwords):,}")

filtered = [p for p in leak_passwords if p not in rockyou_set]

removed = len(leak_passwords) - len(filtered)
print(f"\n  Removed (in RockYou):     {removed:,}")
print(f"  Remaining (filtered):     {len(filtered):,}")

print(f"\nWriting filtered passwords to {OUTPUT_FILE} ...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(filtered) + "\n")

print("Done.")
