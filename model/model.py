import torch
import torch.nn as nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs:
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.):
        super().__init__()
        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = self.attend(qk)
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.):
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'

        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_kv):
        q = self.to_q(x_q)
        kv = self.to_kv(x_kv).chunk(2, dim=-1)

        if q.dim() == 2:
            q = q.unsqueeze(1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = self.attend(qk)
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads=heads, dropout=dropout)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x_q, x_kv):
        for attn, ff in self.layers:
            x_q = attn(x_q, x_kv=x_kv) + x_q
            x_q = ff(x_q) + x_q
        return x_q


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads=heads, dropout=dropout)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class GeneWiseAttentionPooling(nn.Module):
    """Per-gene attention pooling over tile embeddings."""
    def __init__(self, emb_dim, gene_nums, dropout=0.):
        super().__init__()
        self.gene_nums = gene_nums
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, self.gene_nums),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (B, N, C)
        attn_weights = self.attn(x)                        # (B, N, gene_nums)
        attn_weights = attn_weights.permute(0, 2, 1)       # (B, gene_nums, N)
        pooled = (x.unsqueeze(1) * attn_weights.unsqueeze(-1)).sum(dim=2)  # (B, gene_nums, C)
        return pooled, attn_weights


class SIGMABase(nn.Module):
    """Simplified SIGMA: tile encoder + gene-wise pooling + per-gene MLP (without cross-attention fusion)."""
    def __init__(self, gene_nums=250, emb_dim=1536, depth=4, heads=8, mlp_dim=1536, dropout=0.):
        super().__init__()

        self.tiles_encoder = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.pool = GeneWiseAttentionPooling(emb_dim, gene_nums, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, x_spot, x_tiles):
        # x_spot: (B, C), x_tiles: (B, N, C)
        x_tiles_encoded = self.tiles_encoder(x_tiles)      # (B, N, C)
        x_tiles_agg, attn_weights = self.pool(x_tiles_encoded)  # (B, gene_nums, C)
        out = self.mlp(x_tiles_agg).squeeze(-1)             # (B, gene_nums)
        return out, attn_weights


class SIGMA(nn.Module):
    """SIGMA: tile encoder + gene-wise pooling + cross-attention fusion + MLP head."""
    def __init__(self, gene_nums=250, emb_dim=1536, depth=3, heads=8, mlp_dim=1536, dropout=0.):
        super().__init__()

        self.tiles_encoder = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.fusion_encoder = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.pool = GeneWiseAttentionPooling(emb_dim, gene_nums, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, emb_dim), requires_grad=True)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1)
        )

    def forward(self, x_spot, x_tiles):
        # x_spot: (B, C), x_tiles: (B, N, C)
        x_tiles = x_tiles + self.pos_embedding
        x_tiles_encoded = self.tiles_encoder(x_tiles)       # (B, N, C)
        x_tiles_agg, attn_weights = self.pool(x_tiles_encoded)  # (B, gene_nums, C)
        x_patch_fused = self.fusion_encoder(x_tiles_agg, x_spot.unsqueeze(1))  # (B, gene_nums, C)
        x_combined = torch.cat((x_tiles_agg, x_patch_fused), dim=-1)  # (B, gene_nums, C*2)
        out = self.mlp(x_combined).squeeze(-1)               # (B, gene_nums)
        return out, attn_weights
