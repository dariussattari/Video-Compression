from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    hidden_state: Optional[torch.Tensor] = None
    past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0

        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        needs_rebuild = (
            self.cos_cached is None
            or self.sin_cached is None
            or seq_len > self.max_seq_len_cached
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        )
        if not needs_rebuild:
            return

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.cos_cached = freqs.cos().to(dtype=dtype)
        self.sin_cached = freqs.sin().to(dtype=dtype)
        self.max_seq_len_cached = seq_len

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = -x_odd
        out[..., 1::2] = x_even
        return out

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, H, T, D = x.shape
        assert D == self.dim

        max_pos = offset + T
        self._build_cache(max_pos, x.device, x.dtype)

        cos = self.cos_cached[offset:offset + T]
        sin = self.sin_cached[offset:offset + T]

        cos = torch.repeat_interleave(cos, repeats=2, dim=-1).view(1, 1, T, D)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1).view(1, 1, T, D)

        return x * cos + self.rotate_half(x) * sin


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        x: torch.Tensor,
        rope: nn.Module,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        past_len = 0 if past_k is None else past_k.size(-2)

        q = rope(q, offset=past_len)
        k = rope(k, offset=past_len)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
        if past_v is not None:
            v = torch.cat([past_v, v], dim=-2)

        T_q = q.size(-2)
        T_k = k.size(-2)

        if past_len == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T_q == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            q_pos = torch.arange(past_len, past_len + T_q, device=x.device)
            k_pos = torch.arange(T_k, device=x.device)
            allow = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            attn_mask = allow.view(1, 1, T_q, T_k)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        out = self.out_proj(self._merge(out))

        new_k = k if use_cache else None
        new_v = v if use_cache else None
        return out, new_k, new_v


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        bias: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = int((8 / 3) * embed_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, bias: bool = True):
        super().__init__()

        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = MultiHeadCausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
        )
        self.norm2 = nn.RMSNorm(embed_dim)
        self.ffn = SwiGLUFeedForward(embed_dim=embed_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        rope: nn.Module,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn_out, new_k, new_v = self.attn(
            self.norm1(x),
            rope=rope,
            past_k=past_k,
            past_v=past_v,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_k, new_v


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        bias: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = embed_dim // num_heads

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.rope = RotaryPositionalEmbeddings(self.head_dim)

        self.blocks = nn.ModuleList(
            [DecoderBlock(embed_dim=embed_dim, num_heads=num_heads, bias=bias) for _ in range(num_layers)]
        )

        self.final_norm = nn.RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_hidden_layer: Optional[int] = None,
    ) -> ModelOutput:
        if return_hidden_layer is not None:
            if not (0 <= return_hidden_layer < self.num_layers):
                raise ValueError(f"return_hidden_layer must be in [0, {self.num_layers - 1}]")

        x = self.token_embedding(input_ids)

        if past_kvs is None:
            past_kvs = [None] * self.num_layers
        else:
            if len(past_kvs) != self.num_layers:
                raise ValueError(f"past_kvs must have length {self.num_layers}")

        new_past_kvs = [] if use_cache else None
        hidden_state = None

        for i, block in enumerate(self.blocks):
            past = past_kvs[i]
            past_k = None if past is None else past[0]
            past_v = None if past is None else past[1]

            x, new_k, new_v = block(
                x,
                rope=self.rope,
                past_k=past_k,
                past_v=past_v,
                use_cache=use_cache,
            )

            if return_hidden_layer == i:
                hidden_state = x

            if use_cache:
                new_past_kvs.append((new_k, new_v))

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return ModelOutput(
            logits=logits,
            hidden_state=hidden_state,
            past_kvs=new_past_kvs,
        )