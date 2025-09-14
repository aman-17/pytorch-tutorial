import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),    
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(MultiHeadAttention, self).__init__()
        assert dim % heads == 0, "Dimension must be divisible by number of heads."
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim * 3, bias=False)
        self.k = nn.Linear(dim, dim * 3, bias=False)
        self.v = nn.Linear(dim, dim * 3, bias=False)
        self.lm_head = nn.Linear(dim, dim)
    
    def forward(self, x):
        b, seq_len, dim = x.size()
        q = self.q(x).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, seq_len, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, seq_len, dim)
        return self.lm_head(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, depth, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim, heads=heads),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, channels=3, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_dim = channels * patch_size * patch_size
        self.dim = dim

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerBlock(
            dim=dim, heads=heads, depth=depth, mlp_dim=mlp_dim, dropout=dropout
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)
        
    def forward(self, x): # x: image
        b, c, h, w = x.size()
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding[:, :(n+1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)