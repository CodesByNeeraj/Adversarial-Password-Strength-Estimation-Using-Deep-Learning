"""
split_data.py

Reads data/fullData.txt, filters to <= 10 chars, shuffles, then splits 80/20
into data/train.txt and data/test.txt.

Train set keeps passwords as-is (including duplicates).
Test set has any password that appears in train removed.

Usage:
    python split_data.py
"""

import random

INPUT_FILE  = "data/fullData.txt"
TRAIN_FILE  = "data/train.txt"
TEST_FILE   = "data/test.txt"
SEED        = 42
TRAIN_RATIO = 0.80
MAX_LEN     = 10  # must match seq_len in train.py

print(f"Reading {INPUT_FILE} ...")
with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw = [
        line.strip() for line in f
        if line.strip()
        and len(line.strip()) <= MAX_LEN
        and line.strip().isascii()   # drop non-ASCII passwords: reduces vocab from ~576 → ~95
    ]

print(f"Total lines read (<=10 chars, ASCII only) : {len(raw):,}")

# --- Shuffle ---
rng = random.Random(SEED)
rng.shuffle(raw)

# --- Split 80/20 by line count ---
split_idx = int(len(raw) * TRAIN_RATIO)
train = raw[:split_idx]
test  = raw[split_idx:]

print(f"\nSplit ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)})")
print(f"  Train (before dedup) : {len(train):,}")
print(f"  Test  (before dedup) : {len(test):,}")

# --- Remove from test any password that appears in train ---
train_set = set(train)
test_clean = [p for p in test if p not in train_set]

print(f"\n  Test passwords removed (also in train) : {len(test) - len(test_clean):,}")
print(f"  Test  (after dedup)  : {len(test_clean):,}")

# --- Write ---
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(train) + "\n")
print(f"\nSaved train -> {TRAIN_FILE}")

with open(TEST_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(test_clean) + "\n")
print(f"Saved test  -> {TEST_FILE}")
