import torch.nn as nn
import torch
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.seq_length, config.d_model)
    def forward(self, x): # x: (B, seq_length, d_model)
        B, T, C = x.shape
        pos = torch.arange(T, device=x.device)
        embeddings = self.embedding(pos) # (seq_length, d_model)
        embeddings = embeddings.unsqueeze(0) # (1, seq_length, d_model)
        return x + embeddings

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        assert self.d_model % self.n_heads == 0
    def forward(self, x): # x: (B, seq_length, d_model)
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # (B, n_heads, T, d_model / n_heads)
        y = y.transpose(1, 2).view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.d_model * 4, config.d_model)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.sa = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.sa(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.pos_emb = PositionalEmbedding(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)
        self.att_pool = nn.Linear(config.d_model, 1)
    def forward(self, x): # x: (B, seq_length, input_dim)
        x = self.input_proj(x) # (B, seq_length, d_model)
        x = self.pos_emb(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        weights = torch.softmax(self.att_pool(x), dim=1) # (B, T, 1)
        h = (weights * x).sum(dim=1) # (B, d_model)
        x = self.head(h) # (B, 1)
        return x
