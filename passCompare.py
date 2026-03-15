# files
generated_file = "generated_passwords.txt"
full_file = "data/fullData.txt"

# -------------------------------
# Step 1: load generated dataset
# -------------------------------
with open(generated_file, "r", encoding="utf-8") as f:
    generated_set = set(line.strip() for line in f)

print("Generated dataset size:", len(generated_set))

# -----------------------------------------
# Step 2: read full dataset (skip first 900K)
# -----------------------------------------
matches = 0
checked = 0

with open(full_file, "r", encoding="utf-8") as f:
    
    # skip first 900k lines
    for _ in range(900000):
        next(f)

    # compare remaining lines
    for line in f:
        checked += 1
        item = line.strip()

        if item in generated_set:
            matches += 1

# -----------------------------------------
# Results
# -----------------------------------------
print("Lines checked after skipping:", checked)
print("Matches found:", matches)

similarity = matches / len(generated_set) * 100
print(f"Similarity percentage: {similarity:.4f}%")