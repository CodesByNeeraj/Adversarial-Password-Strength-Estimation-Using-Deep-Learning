import json
import torch
from torch.utils.data import Dataset

MASK_TOKEN = '[MASK]'
PAD_TOKEN  = '[PAD]'
MAX_LEN    = 10


class PasswordVocab:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.mask_id = 0
        self.pad_id  = 0

    def build(self, passwords):
        special = [PAD_TOKEN, MASK_TOKEN]
        chars   = sorted(set(ch for p in passwords for ch in p))
        tokens  = special + chars
        self.char2idx  = {c: i for i, c in enumerate(tokens)}
        self.idx2char  = {i: c for i, c in enumerate(tokens)}
        self.vocab_size = len(tokens)
        self.pad_id    = self.char2idx[PAD_TOKEN]
        self.mask_id   = self.char2idx[MASK_TOKEN]

    def encode(self, password):
        ids = [self.char2idx.get(ch, self.pad_id) for ch in password[:MAX_LEN]]
        ids += [self.pad_id] * (MAX_LEN - len(ids))
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            c = self.idx2char.get(i, '')
            if c in (PAD_TOKEN, MASK_TOKEN, ''):
                continue
            out.append(c)
        return ''.join(out)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': {str(k): v for k, v in self.idx2char.items()},
            }, f)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.char2idx   = data['char2idx']
        self.idx2char   = {int(k): v for k, v in data['idx2char'].items()}
        self.vocab_size = len(self.char2idx)
        self.pad_id     = self.char2idx[PAD_TOKEN]
        self.mask_id    = self.char2idx[MASK_TOKEN]


class PasswordDataset(Dataset):
    def __init__(self, file_path, vocab):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            passwords = [
                line.strip() for line in f
                if line.strip()
                and len(line.strip()) <= MAX_LEN
                and line.strip().isascii()
            ]
        self.data = [
            torch.tensor(vocab.encode(p), dtype=torch.long)
            for p in passwords
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
