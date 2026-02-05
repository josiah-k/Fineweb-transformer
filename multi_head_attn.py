import torch
import torch.nn as nn
import math

"""
Todo:
- add code for inference mode
    - set up KV cache properly per layer
    - add casual mask just for initial prompt, remove otherwise
- add scalable softmax (look at paper)
"""


class Attn_head(nn.Module):
    def __init__(self, hidden_dim: int,
                num_heads: int,
                head_dim: int,
                similarity: str = 'softmax'):
        super().__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.similarity = similarity

    def forward(self, x: torch.tensor, kv_cache: torch.tensor, pad_mask: torch.tensor):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x) # (B, seq_len, hidden_dim)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            K, V = torch.cat([past_k, K], dim=1), torch.cat([past_v, V], dim=1)
        K_cache, V_cache = K.detach(), V.detach()

        # reshape to expose heads
        batch_dim = x.shape[0]
        Q = Q.view(batch_dim, -1, self.num_heads, self.head_dim) # (B, seq_len, num_heads, head_dim)
        K = K.view(batch_dim, -1, self.num_heads, self.head_dim)
        V = V.view(batch_dim, -1, self.num_heads, self.head_dim)

        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2) # (B, num_heads, seq_len, head_dim)

        scores = Q @ K.transpose(-2,-1) / math.sqrt(self.head_dim)

        attn_mat = self.compute_masked_attn(scores, pad_mask)

        output = attn_mat @ V
        # concatenate heads back together
        output = output.transpose(1,2) # (B, seq_len, num_heads, head_dim)
        output = output.contiguous().view(batch_dim, -1, self.hidden_dim) # (B, seq_len, hidden_dim)

        return output, (K_cache, V_cache)

    def compute_masked_attn(self, scores: torch.tensor, pad_mask: torch.tensor):
        """
        pad_mask assumes 1 == include, 0 == mask
        """
        pad_mask = pad_mask.unsqueeze(1).expand(-1,pad_mask.shape[1],-1) # expand to match shape of causal mask
        pad_mask = pad_mask.unsqueeze(1).expand(-1,self.num_heads,-1,-1)

        if self.similarity == 'softmax':
            causal_mask = torch.triu(torch.ones_like(scores), diagonal=1)
            pad_mask = 1 - pad_mask
            mask = (causal_mask + pad_mask)*-1e6
            attn_mat = torch.softmax(scores + mask, -1)

        elif self.similarity == 'linear':
            causal_mask = torch.tril(torch.ones_like(scores))
            mask = causal_mask + pad_mask
            attn_mat = scores * mask / (scores * mask).sum(-1, keepdim=True)

        return attn_mat


class Multi_head_attn(nn.Module):
    def __init__(self, n_heads: int, hidden_dim: int, head_dim: int, similarity: str):
        super().__init__()

        self.n_heads = n_heads
        self.Wo = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.attn_heads = Attn_head(hidden_dim, n_heads, head_dim, similarity=similarity)

    def forward(self, x: torch.tensor, kv_cache: torch.tensor, pad_mask: torch.tensor):
        heads_out, _ = self.attn_heads(x, kv_cache, pad_mask)
        output = heads_out @ self.Wo
        """
        currently kv_cache is returned but not used, need to set up inference mode correctly
        """

        return output


class FFN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.W1, self.W2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim)), nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b1, self.b2 = nn.Parameter(torch.randn(hidden_dim)), nn.Parameter(torch.randn(hidden_dim))
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor):
        x = self.relu(x @ self.W1 + self.b1)
        x = x @ self.W2 + self.b2
        return x


class Pos_embed(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int):
        super().__init__()

        self.P = torch.zeros((1, max_len, hidden_dim))
        X = torch.arange(max_len).reshape(-1,1) / torch.pow(10000, torch.arange(0, hidden_dim, 2) / hidden_dim)
        self.P[:,:, 0::2] = torch.sin(X)
        self.P[:,:, 1::2] = torch.cos(X)

    def forward(self, x: torch.tensor):
        return x + self.P[:, :x.shape[1], :]


class Transformer_layer(nn.Module):
    def __init__(self, n_heads: int, hidden_dim: int, similarity: str):
        super().__init__()

        assert hidden_dim%n_heads == 0
        head_dim = hidden_dim // n_heads

        self.multi_head_attn  = Multi_head_attn(n_heads, hidden_dim, head_dim, similarity)
        self.ffn = FFN(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, data: dict):
        x, kv_cache, pad_mask = data['x'], data['kv_cache'], data['pad_mask']

        x = x + self.multi_head_attn(x, kv_cache, pad_mask)
        x = self.layer_norm(x)
        x = x + self.ffn(x)
        x = self.layer_norm(x)

        return {'x': x,
                'kv_cache': kv_cache,
                'pad_mask': pad_mask}


class Transformer(nn.Module):
    def __init__(self, n_heads: int,
                 n_layers: int,
                 hidden_dim: int,
                 max_len: int,
                 vocab_size: int,
                 similarity: str = 'softmax'):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = Pos_embed(hidden_dim, max_len)
        self.transformer = nn.Sequential(*[Transformer_layer(n_heads, hidden_dim, similarity) for _ in range(n_layers)])
        self.W_decode = nn.Parameter(torch.randn(hidden_dim, vocab_size)) # assuming output of (seq_len, hidden_dim)

    def forward(self, x: torch.tensor, pad_mask: torch.tensor, kv_cache = None):

        x = self.embedding(x)
        x = self.pos_embed(x)

        data = {'x': x,
                'kv_cache': kv_cache,
                'pad_mask': pad_mask}
        data = self.transformer(data)
        x = data['x']

        logits = x @ self.W_decode

        return logits
