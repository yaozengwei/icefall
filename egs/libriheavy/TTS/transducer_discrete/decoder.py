# Copyright    2023  Xiaomi Corp.        (authors: Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import math
from typing import Optional

import torch
from torch import Tensor, nn
from icefall.utils import make_pad_mask

from encoder import MultiHeadAttention


class Transformer(nn.Module):
    """
    Args:
        vocab_size: Number of tokens as modeling units including blank.
        sos_id: ID of the SOS symbol.
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module.
        num_heads: number of parallel attention heads.
        feedforward_dim: hidden dimention in the feedforward module
        num_layers: number of encoder layers
        dropout: dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        sos_id: int,
        embed_dim: int = 512,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.sos_id = sos_id

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        self.pos = PositionalEncoding(embed_dim, dropout)

        decoder_layer = TransformerDecoderLayer(
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

    def forward(self, x: Tensor, x_lens: Tensor) -> Tensor:
        """
        Args:
            x: Input token ids of shape (batch, seq_len).
            x_lens: A tensor of shape (batch,) containing the number of tokens in `x`
              before padding.

        Outputs:
            Output tensor of shape (batch, tgt_len, embed_dim).
        """
        batch, seq_len = x.shape

        x = self.embed(x)  # (batch, seq_len, embed_dim)
        x = self.pos(x)

        x = x.permute(1, 0, 2)  # (seq_len, batch, embed_dim)

        padding_mask = make_pad_mask(x_lens)  # (batch, seq_len)
        causal_mask = subsequent_mask(seq_len, device=x.device)  # (seq_len, seq_len)
        attn_mask = torch.logical_or(
            padding_mask.unsqueeze(1),  # (batch, 1, seq_len)
            torch.logical_not(causal_mask).unsqueeze(0)  # (1, seq_len, seq_len)
        )  # (batch, seq_len, seq_len)

        x = self.decoder(x, attn_mask=attn_mask)

        x = x.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    TransformerDecoderLayer is made up of self-attention and feedforward.

    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module.
        num_heads: number of parallel attention heads.
        feedforward_dim: hidden dimention in the feedforward module
        dropout: dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(
            embed_dim, attention_dim, num_heads, dropout=0)
        self.norm_self_attn = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm_ff = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input sequence of shape (seq_len, batch, embed_dim).
            attn_mask: A binary mask for self-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
        """
        # self-attention module
        qkv = self.norm_self_attn(x)
        attn_out = self.self_attn(query=qkv, key=qkv, value=qkv, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # feed-forward module
        ff_out = self.feed_forward(self.norm_ff(x))
        x = x + self.dropout(ff_out)

        return x


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer class.
        num_layers: number of decoder layers.
    """

    def __init__(self, decoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Input sequence of shape (seq_len, batch, embed_dim).
            attn_mask: A binary mask for self-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
        """
        output = x

        for layer_index, mod in enumerate(self.layers):
            output = mod(output, attn_mask=attn_mask)

        return output


class PositionalEncoding(nn.Module):
    """Positional encoding.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py.

    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(self, embed_dim, dropout_rate, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.xscale = math.sqrt(embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.embed_dim)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, embed_dim)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor):
        """Add positional encoding.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Encoded tensor of shape (batch, seq_len, embed_dim).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.dropout(self.pe[:, :x.size(1)])
        return x


# from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/mask.py
def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create causal mask (size, size).

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)


def _test_transformer():
    embed_dim = 512
    batch_size = 5
    seq_len = 100
    vocab_size = 1025

    x = torch.randint(vocab_size, (batch_size, seq_len))
    x_lens = torch.full((batch_size,), seq_len)

    m = Transformer(vocab_size=vocab_size, blank_id=1024, embed_dim=embed_dim)
    y = m(x, x_lens)

    assert y.shape == (batch_size, seq_len, embed_dim), y.shape
    print("passed")


if __name__ == "__main__":
    _test_transformer()
