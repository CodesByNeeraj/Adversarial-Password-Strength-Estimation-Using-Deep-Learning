import torch
import string
from models import Generator

# 1. Recreate the exact same character map from your Dataset
chars = "`" + string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^"
idx2char = {idx: char for idx, char in enumerate(chars)}

# 2. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 10
vocab_size = 69
hidden_dim = 128

def generate_passwords(num_to_generate=10):
    print("Loading Generator...")
    # Initialize the model skeleton
    gen = Generator(seq_len, vocab_size, hidden_dim).to(device)
    
    # Load the trained weights
    gen.load_state_dict(torch.load("generator_weights.pth", map_location=device))
    
    # Put the model in "evaluation" mode (turns off training features)
    gen.eval()
    
    # 3. Create random noise (the "seed" for our passwords)
    noise = torch.randn(num_to_generate, 128).to(device)
    
    # 4. Generate!
    with torch.no_grad(): # Don't track gradients, saves memory
        # Output shape: (batch_size, 69_vocab, 10_positions)
        raw_output = gen(noise) 
        
    # 5. Find the highest probability character for each position
    # This squashes the 69 probabilities down to 1 winning index
    best_guesses = torch.argmax(raw_output, dim=1) 
    
    # 6. Translate numbers back to text
    generated_passwords = []
    for row in best_guesses:
        pwd = ""
        for idx in row:
            pwd += idx2char[idx.item()]
            
        # Strip away the padding characters to get the real password
        clean_pwd = pwd.replace("`", "") 
        generated_passwords.append(clean_pwd)
        
    return generated_passwords

if __name__ == "__main__":
    passwords = generate_passwords(20)
    
    print("\n--- GENERATED PASSWORDS ---")
    for p in passwords:
        print(p)