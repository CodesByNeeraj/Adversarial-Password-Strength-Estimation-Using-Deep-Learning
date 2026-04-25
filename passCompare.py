generated_file = "generated/gpt2_generated.txt"
test_file = "data/test.txt"


#step 1: Load generated passwords (remove duplicates)
with open(generated_file, "r", encoding="utf-8") as f:
    generated_set = set(line.strip() for line in f if line.strip())

print("Unique passwords in generated password file:", len(generated_set))


#step 2: Load held-out test set (already deduplicated and guaranteed not to overlap with train.txt)
with open(test_file, "r", encoding="utf-8") as f:
    test_set = set(line.strip() for line in f if line.strip())

print("Unique passwords in test set:", len(test_set))


#step 3:compute unique matches
matches = generated_set.intersection(test_set)

print("Unique matches:", len(matches))


#step 4:similarity percentage
similarity = (len(matches) / len(test_set)) * 100

print(f"Match rate (matches / test_set, PassGAN metric): {similarity:.4f}%")
