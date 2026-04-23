import json
import torch
from models import Generator

#Load vocabulary built during training
with open("vocab.json", "r", encoding="utf-8") as f:
    idx2char_list = json.load(f)
idx2char = {idx: char for idx, char in enumerate(idx2char_list)}

#Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 10
vocab_size = len(idx2char_list)
hidden_dim = 128

def generate_passwords(num_to_generate=10, batch_size=1024):
    print("Loading Generator...")
    gen = Generator(seq_len, vocab_size, hidden_dim).to(device)
    gen.load_state_dict(torch.load("generator_weights.pth", map_location=device))
    gen.eval()

    generated_passwords = []
    generated = 0

    with torch.no_grad():
        while generated < num_to_generate:
            current_batch = min(batch_size, num_to_generate - generated)
            noise = torch.randn(current_batch, 128).to(device)
            raw_output = gen(noise)
            best_guesses = torch.argmax(raw_output, dim=1)

            for row in best_guesses:
                pwd = "".join(idx2char[idx.item()] for idx in row)
                clean_pwd = pwd.replace(idx2char[0], "")
                generated_passwords.append(clean_pwd)

            generated += current_batch
            if generated % 10000 == 0 or generated == num_to_generate:
                print(f"Generated {generated}/{num_to_generate}...")

    return generated_passwords
'''
if __name__ == "__main__":
    passwords = generate_passwords(100000)
    
    print("\n--- GENERATED PASSWORDS ---")
    for p in passwords:
        print(p)
'''

#modify as needed (to generate different amounts of passwords)
if __name__ == "__main__":
    num_generate = 1000000
    passwords = generate_passwords(num_generate)

    output_file = "generated_1e6.txt"

    print(f"\nSaving {num_generate} passwords to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for p in passwords:
            f.write(p + "\n")

    print("Done.")
