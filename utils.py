from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.init as init
import string
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_data_path, seq_len=10):
        self.seq_len = seq_len
        
        #69 allowed characters
        self.chars = "`" + string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^"
        
        # Creating a dictionary mapping characters to numbers (e.g., 'a' -> 1, 'b' -> 2)
        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        
        # 2. Reading the file
        print(f"Loading data from {text_data_path}...")
        with open(text_data_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read all lines, strip whitespace, and ignore empty lines
            self.passwords = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(self.passwords)} passwords.")

    def __len__(self):
        return len(self.passwords)
    
    def __getitem__(self, idx):
        password = self.passwords[idx]
        
        # 3. Truncate if too long
        if len(password) > self.seq_len:
            password = password[:self.seq_len]
            
        # 4. Pad if too short using the backtick '`' (which is index 0)
        password = password.ljust(self.seq_len, "`")
        
        # 5. Convert characters to indices
        # If a weird character isn't in our 69-char list, default to 0 to prevent crashes
        indices = [self.char2idx.get(char, 0) for char in password]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m,(nn.Conv1d,nn.Linear)):
                init.kaiming_uniform_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias,0)
            
    
