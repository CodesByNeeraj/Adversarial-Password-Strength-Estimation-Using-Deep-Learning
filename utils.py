
import collections
import json
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_data_path, seq_len=10):
        self.seq_len = seq_len

        # 1. Read file, filter passwords that are too long
        print(f"Loading data from {text_data_path}...")
        with open(text_data_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.passwords = [line.strip() for line in f if line.strip() and len(line.strip()) <= seq_len]
        print(f"Loaded {len(self.passwords)} passwords.")

        # build vocabulary dynamically from data (like original PassGAN)
        # Index 0 is reserved for 'unk' (unknown / padding)
        counts = collections.Counter(char for pwd in self.passwords for char in pwd)

        self.char2idx = {'unk': 0}
        self.idx2char = ['unk']

        for char, _ in counts.most_common():
            self.char2idx[char] = len(self.idx2char)
            self.idx2char.append(char)

        self.vocab_size = len(self.idx2char)
        print(f"Vocabulary size: {self.vocab_size} unique characters.")

    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.idx2char, f, ensure_ascii=False)
        print(f"Vocab saved to {path}")

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        password = self.passwords[idx]

        # Convert characters to indices; pad positions and unknown chars map to 0 (unk)
        indices = [self.char2idx.get(char, 0) for char in password]
        # Pad to seq_len with 0 (unk index)
        indices += [0] * (self.seq_len - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
