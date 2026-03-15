# files
generated_file = "generated_passwords.txt"
full_file = "data/fullData.txt"
# --------------------------------------------------
# Step 1: Load generated passwords (remove duplicates)
# --------------------------------------------------
with open(generated_file, "r", encoding="utf-8") as f:
    generated_set = set(line.strip() for line in f if line.strip())

print("Unique passwords in generated.txt:", len(generated_set))

# --------------------------------------------------
# Step 2: Read full dataset, skip first 900K, remove duplicates
# --------------------------------------------------
full_set = set()

with open(full_file, "r", encoding="utf-8") as f:
    
    # Skip first 900k lines
    for _ in range(900000):
        next(f)

    # Store remaining unique passwords
    for line in f:
        pwd = line.strip()
        if pwd:
            full_set.add(pwd)

print("Unique passwords in filtered full dataset:", len(full_set))

# --------------------------------------------------
# Step 3: Compute unique matches
# --------------------------------------------------
matches = generated_set.intersection(full_set)

print("Unique matches:", len(matches))

# --------------------------------------------------
# Step 4: Similarity percentage
# --------------------------------------------------
similarity = (len(matches) / len(generated_set)) * 100

print(f"Unique similarity: {similarity:.4f}%")
