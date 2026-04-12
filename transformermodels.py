import torch
import torch.nn as nn

#vocab_size --> how many unique characteristics exist
#seq_len ---> fixed length of every password sequence
#d_model=128 --> embedding size
#nhead = 4 --> number of attention heads
#num_layers = 4 ---> how many stacked transformer blocks
class PasswordTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=128, nhead=4, num_layers=4):
        super().__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

        # learned positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # register_buffer so the mask automatically moves to the correct device
        # when model.to(device) is called
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        self.register_buffer('mask', mask)

        #1 transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )

        #4 stacks of encoder layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, current_len)
        t = x.size(1)
        # (batch, t, d_model)
        x = self.embedding(x)   
        # add positional info                        
        x = x + self.pos_embedding[:, :t, :]           

        # slice causal mask to current sequence length (important during generation
        # when we grow the sequence one token at a time)
        mask = self.mask[:t, :t]

        x = self.transformer(x, mask=mask)
        # (batch, t, vocab_size)
        logits = self.fc(x)                           

        return logits

    @torch.no_grad()
    def generate(self, num_passwords, device, temperature=1.0, unk_idx=0):
        """
        Autoregressively sample passwords.
        Uses unk (index 0) as a BOS token, generates seq_len more tokens,
        then strips the BOS so output length is seq_len.
        """
        self.eval()

        # seed with BOS = unk token
        tokens = torch.full((num_passwords, 1), unk_idx, dtype=torch.long, device=device)

        for _ in range(self.seq_len - 1):
            # (batch, current_len, vocab_size)
            logits = self.forward(tokens)        
            # only care about the last position
            next_logits = logits[:, -1, :]       
            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            # (batch, 1)
            next_token = torch.multinomial(probs, num_samples=1) 
            tokens = torch.cat([tokens, next_token], dim=1)

        # strip the BOS token; result is (num_passwords, seq_len - 1)
        return tokens[:, 1:]