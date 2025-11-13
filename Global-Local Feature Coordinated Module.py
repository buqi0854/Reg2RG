import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Type

class MultiLayerPerceptron(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            activation: nn.Module = nn.ReLU,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class BidirectionalAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()

        self.image_to_region_cross_attn = AttentionLayer(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mlp = MultiLayerPerceptron(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.region_to_image_cross_attn = AttentionLayer(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
            self, queries: Tensor, keys: Tensor
    ) -> Tuple[Tensor, Tensor]:
        q = queries
        k = keys
        attn_out = self.image_to_region_cross_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm1(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm2(queries)

        return queries, keys


class AttentionLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.0,
            kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
                self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_into_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_from_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_into_heads(q, self.num_heads)
        k = self._separate_into_heads(k, self.num_heads)
        v = self._separate_into_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_from_heads(out)
        out = self.out_proj(out)

        return out