import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    #timestep embedding: maps scalar t -> vector of size dim.
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class DiffusionTransformer(nn.Module):

    def __init__(self, vocab_size, hidden_dim=256, n_heads=8,
                 n_layers=6, seq_len=10):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb   = nn.Embedding(seq_len, hidden_dim)
        self.time_mlp  = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            #pre-norm: more stable training
            norm_first=True,  
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj    = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_t, t):
        B, L   = x_t.shape
        pos    = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        #(B, 1, D)
        t_emb  = self.time_mlp(t).unsqueeze(1)  
        #(B, L, D)    
        h      = self.token_emb(x_t) + self.pos_emb(pos) 
        #broadcast over seq
        h      = h + t_emb
        #(B, L, D)                               
        h      = self.transformer(h) 
        #(B, L, vocab_size)                   
        return self.out_proj(h)                           
