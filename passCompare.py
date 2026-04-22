# files
#(base) PS Microsoft.PowerShell.Core\FileSystem::\\wsl.localhost\Ubuntu\home\raa524\DL\cloned\Adversarial-Password-Strength-Estimation-Using-Deep-Learning> & "C:\Users\rajne\passganenv\Scripts\Activate.ps1"
#>> python passCompare.py 
#
generated_file = "gpt2_generated.txt"
test_file = "data/test.txt"

# --------------------------------------------------
# Step 1: Load generated passwords (remove duplicates)
# --------------------------------------------------
with open(generated_file, "r", encoding="utf-8") as f:
    generated_set = set(line.strip() for line in f if line.strip())

print("Unique passwords in generated password file:", len(generated_set))

# --------------------------------------------------
# Step 2: Load held-out test set (already deduplicated
#         and guaranteed not to overlap with train.txt)
# --------------------------------------------------
with open(test_file, "r", encoding="utf-8") as f:
    test_set = set(line.strip() for line in f if line.strip())

print("Unique passwords in test set:", len(test_set))

# --------------------------------------------------
# Step 3: Compute unique matches
# --------------------------------------------------
matches = generated_set.intersection(test_set)

print("Unique matches:", len(matches))

# --------------------------------------------------
# Step 4: Similarity percentage
# --------------------------------------------------
similarity = (len(matches) / len(test_set)) * 100

print(f"Match rate (matches / test_set, PassGAN metric): {similarity:.4f}%")
