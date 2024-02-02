#!/usr/bin/env python3
# Copyright           2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
import logging
import math
import random
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from icefall.utils import make_pad_mask
from scaling import penalize_abs_values_gt


class Conformer(nn.Module):
    """
    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module.
        num_heads: number of parallel attention heads.
        feedforward_dim: hidden dimention in the feedforward module
        cnn_module_kernel: convolution kernel size
        num_layers: number of encoder layers
        memory_dim: dimension of memory embedding, optional.
        dropout: dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 512,
        attention_dim: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        cnn_module_kernel: int = 5,
        num_layers: int = 6,
        memory_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder_pos = RelPositionalEncoding(embed_dim, dropout)

        encoder_layer = ConformerEncoderLayer(
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            cnn_module_kernel=cnn_module_kernel,
            memory_dim=memory_dim,
            dropout=dropout,
        )
        self.encoder = ConformerEncoder(encoder_layer, num_layers)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        memory: Optional[Tensor] = None,
        memory_lens: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          x:
            Input tensor of shape (batch, tgt_len, embed_dim).
          x_lens:
            A tensor of shape (batch,) containing the number of frames in
            `x` before padding.
          memory:
            Memory sequence of shape (batch, src_len, memory_dim).
          memory_lens:
            A tensor of shape (batch,) containing the number of frames in
            `memory` before padding.

        Outputs:
            Output tensor of shape (batch, tgt_len, embed_dim).
        """
        padding_mask = make_pad_mask(x_lens)  # (batch, tgt_len)

        if memory is not None:
            memory_padding_mask = make_pad_mask(memory_lens)
            memory_attn_mask = torch.logical_or(
                padding_mask.unsqueeze(2),  # (batch, tgt_len, 1)
                memory_padding_mask.unsqueeze(1),  # (batch, 1, src_len)
            )  # (batch, tgt_len, src_len)
            memory = memory.permute(1, 0, 2)  # (src_len, batch, memory_dim)
        else:
            memory_attn_mask = None

        pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (tgt_len, batch, embed_dim)

        x = self.encoder(
            x,
            pos_emb=pos_emb,
            padding_mask=padding_mask,
            memory=memory,
            memory_attn_mask=memory_attn_mask,
        )

        x = self.norm(x)

        x = x.permute(1, 0, 2)  # (batch, tgt_len, embed_dim)
        return x


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of
        feedforward, self-attention, (optional cross-attention), convolution, feedforward.

    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module.
        num_heads: number of parallel attention heads.
        feedforward_dim: hidden dimention in the feedforward module
        cnn_module_kernel: convolution kernel size
        memory_dim: dimension of memory embedding, optional.
        dropout: dropout rate
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        feedforward_dim: int,
        cnn_module_kernel: int,
        memory_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()

        self.feed_forward_macaron = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm_ff_macaron = nn.LayerNorm(embed_dim)

        self.self_attn = RelPositionMultiHeadAttention(
            embed_dim, attention_dim, num_heads, dropout=0)
        self.norm_self_attn = nn.LayerNorm(embed_dim)

        if memory_dim is not None:
            self.cross_attn = MultiHeadAttention(
                embed_dim, attention_dim, num_heads,
                memory_dim=memory_dim, dropout=dropout)
            self.norm_cross_attn_q = nn.LayerNorm(embed_dim)
            self.norm_cross_attn_kv = nn.LayerNorm(memory_dim)

        self.conv_module = ConvolutionModule(embed_dim, cnn_module_kernel)
        self.norm_conv = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.norm_ff = nn.LayerNorm(embed_dim)

        self.norm_final = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input sequence of shape (seq_len, batch, embed_dim).
            pos_emb: Relative positional embedding tensor of shape (1, seq_len*2-1, embed_dim).
            padding_mask: A binary mask indicating which elements are padding.
                Its shape is (batch, seq_len).
            attn_mask: A binary mask for self-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
            memory: Memory sequence of shape (seq_len, batch, memory_dim).
            memory_attn_mask: A binary mask for cross-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
        """
        # Note: could remove the ff_scale when using ScaledAdam
        ff_scale = 0.5
        # macaron style feed-forward module
        ff_out = self.feed_forward_macaron(self.norm_ff_macaron(x))
        x = x + ff_scale * self.dropout(ff_out)

        # self-attention module
        attn_out = self.self_attn(
            self.norm_self_attn(x), pos_emb=pos_emb, attn_mask=attn_mask,
        )
        x = x + self.dropout(attn_out)

        if memory is not None:
            # multi-head cross-attention module
            q = self.norm_cross_attn_q(x)
            kv = self.norm_cross_attn_kv(memory)
            attn_out = self.cross_attn(
                query=q, key=kv, value=kv, attn_mask=memory_attn_mask,
            )
            x = x + self.dropout(attn_out)

        # convolution module
        conv_out = self.conv_module(self.norm_conv(x), padding_mask=padding_mask)
        x = x + self.dropout(conv_out)

        # feed-forward module
        ff_out = self.feed_forward(self.norm_ff(x))
        x = x + self.dropout(ff_out)

        x = self.norm_final(x)

        return x


class ConformerEncoder(nn.Module):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ConformerEncoderLayer class.
        num_layers: number of encoder layers.
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input sequence of shape (seq_len, batch, embed_dim).
            pos_emb: Relative positional embedding tensor of shape (1, seq_len*2-1, embed_dim).
            padding_mask: A binary mask indicating which elements are padding.
                Its shape is (batch, seq_len).
            attn_mask: A binary mask for self-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
            memory: Memory sequence of shape (seq_len, batch, memory_dim).
            memory_attn_mask: A binary mask for cross-attention module indicating which
                elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).
        """
        output = x

        for layer_index, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                padding_mask=padding_mask,
                attn_mask=attn_mask,
                memory=memory,
                memory_attn_mask=memory_attn_mask,
            )

        return output


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, embed_dim: int, dropout_rate: float, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()

        self.embed_dim = embed_dim
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        x_size = x.size(1)
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x_size * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size, self.embed_dim)
        pe_negative = torch.zeros(x_size, self.embed_dim)
        position = torch.arange(0, x_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.embed_dim)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Embedding tensor of shape (1, 2*seq_len-1, embed_dim).
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(1),
        ]
        return self.dropout(pos_emb)


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-Head Attention layer with relative position encoding

    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module, but must be a multiple of num_heads.
        num_heads: number of parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super(RelPositionMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        assert self.head_dim * num_heads == attention_dim, (
            self.head_dim, num_heads, attention_dim
        )
        self.dropout = dropout
        self.name = None  # will be overwritten in training code; for diagnostics.

        self.in_proj = nn.Linear(embed_dim, 3 * attention_dim, bias=True)
        self.out_proj = nn.Linear(attention_dim, embed_dim, bias=True)

        # linear transformation for positional encoding.
        self.linear_pos = nn.Linear(embed_dim, attention_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: Tensor) -> Tensor:
        """Compute relative positional encoding.

        Args:
            x: Input tensor (batch, num_heads, seq_len, 2*seq_len-1).

        Returns:
            Tensor: tensor of shape (batch, num_heads, seq_len, seq_len)
        """
        (batch_size, num_heads, seq_len, n) = x.shape
        assert n == 2 * seq_len - 1, (n, seq_len)

        batch_stride, head_stride, time_stride, n_stride = x.stride()
        return x.as_strided(
            size=(batch_size, num_heads, seq_len, seq_len),
            stride=(batch_stride, head_stride, time_stride - n_stride, n_stride),
            storage_offset=n_stride * (seq_len - 1),
        )

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (seq_len, batch, embed_dim)
            pos_emb: Positional embedding tensor of shape (1, 2*seq_len-1, pos_dim)
            key_padding_mask: A binary mask indicating which elements are padding.
                Its shape is (batch, seq_len).
            attn_mask: A binary mask indicating which elements will be filled with -inf.
                Its shape is (batch, 1, seq_len) or (batch, seq_len, seq_len).

        Outputs:
            Output tensor of shape (seq_len, batch_size, embed_dim).
        """
        head_dim = self.head_dim
        num_heads = self.num_heads

        seq_len, batch, _ = x.shape

        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        q = q.contiguous().view(seq_len, batch, num_heads, head_dim)
        k = k.contiguous().view(seq_len, batch, num_heads, head_dim)
        v = v.contiguous().view(seq_len, batch * num_heads, head_dim).transpose(0, 1)

        q = q.transpose(0, 1)  # (batch, seq_len, num_head, head_dim)

        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, num_heads, head_dim)
        # (1, 2*seq_len, num_head, head_dim) -> (1, num_head, head_dim, 2*seq_len-1)
        p = p.permute(0, 2, 3, 1)

        # (batch, num_head, seq_len, head_dim)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, num_head, head_dim, seq_len)
        matrix_ac = torch.matmul(q_with_bias_u, k)  # (batch, num_head, seq_len, seq_len)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p)  # (batch, num_head, seq_len, 2*seq_len-1)
        matrix_bd = self.rel_shift(matrix_bd)  # (batch, num_head, seq_len, seq_len)

        # Note: could remove the scaling operation when using ScaledAdam
        # (batch, num_head, seq_len, seq_len)
        attn_weights = (matrix_ac + matrix_bd) / math.sqrt(head_dim)

        # From zipformer.py:
        # This is a harder way of limiting the attention scores to not be too large.
        # It incurs a penalty if any of them has an absolute value greater than 50.0.
        # this should be outside the normal range of the attention scores.  We use
        # this mechanism instead of, say, a limit on entropy, because once the entropy
        # gets very small gradients through the softmax can become very small, and
        # some mechanisms like that become ineffective.
        attn_weights = penalize_abs_values_gt(attn_weights, limit=50.0, penalty=1.0e-04)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch, seq_len), key_padding_mask.shape
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"),
            )

        if attn_mask is not None:
            assert (
                attn_mask.shape == (batch, seq_len, seq_len)
                or attn_mask.shape == (batch, 1, seq_len)
            ), attn_mask.shape
            attn_weights = attn_weights.masked_fill(
                attn_mask.unsqueeze(1), float("-inf"),
            )

        attn_weights = attn_weights.view(batch * num_heads, seq_len, seq_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if random.random() < 0.001:
            print_attn_entropy(attn_weights, self.name)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # (batch_size * num_head, seq_len, head_dim)
        attn_output = torch.bmm(attn_weights, v)
        assert attn_output.shape == (batch * num_heads, seq_len, head_dim), attn_output.shape

        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(seq_len, batch, num_heads * head_dim)

        # (seq_len, batch, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        embed_dim: total dimension of the model.
        attention_dim: dimension in the attention module, but must be a multiple of num_heads.
        num_heads: number of parallel attention heads.
        memory_dim: dimension of memory embedding, optional.
        dropout: a Dropout layer on attn_output_weights.
    """

    def __init__(
        self,
        embed_dim: int,
        attention_dim: int,
        num_heads: int,
        memory_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        assert self.head_dim * num_heads == attention_dim, (
            self.head_dim, num_heads, attention_dim
        )
        self.dropout = dropout
        self.name = None  # will be overwritten in training code; for diagnostics.

        self.linear_q = nn.Linear(embed_dim, attention_dim, bias=True)
        self.linear_k = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )
        self.linear_v = nn.Linear(
            embed_dim if memory_dim is None else memory_dim, attention_dim, bias=True
        )

        self.out_proj = nn.Linear(attention_dim, embed_dim, bias=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute dot product attention.

        Args:
            query: Query tensor of shape (tgt_len, batch, embed_dim).
            key: Key tensor of shape (src_len, batch, embed_dim or memory_dim).
            value: Value tensor of shape (src_len, batch, embed_dim or memory_dim).
            attn_mask: A binary mask indicating which elements will be filled with -inf.
                Its shape is (batch, 1, src_len) or (batch, tgt_len, src_len).

        Returns:
            Output tensor of shape (tgt_len, batch, embed_dim).
        """
        num_heads = self.num_heads
        head_dim = self.head_dim

        tgt_len, batch, _ = query.shape
        src_len = key.shape[0]

        q = self.linear_q(query)  # (tgt_len, batch, num_heads * head_dim)
        k = self.linear_k(key)  # (src_len, batch, num_heads * head_dim)
        v = self.linear_v(value)  # (src_len, batch, num_heads * head_dim)

        q = q.reshape(tgt_len, batch, num_heads, head_dim)
        q = q.permute(1, 2, 0, 3)  # (batch, head, tgt_len, head_dim)
        k = k.reshape(src_len, batch, num_heads, head_dim)
        k = k.permute(1, 2, 3, 0)  # (batch, head, head_dim, src_len)
        v = v.reshape(src_len, batch, num_heads, head_dim)
        v = v.reshape(src_len, batch * num_heads, head_dim).transpose(0, 1)

        # Note: could remove the scaling operation when using ScaledAdam
        # (batch, head, tgt_len, src_len)
        attn_weights = torch.matmul(q, k) / math.sqrt(head_dim)

        # From zipformer.py:
        # This is a harder way of limiting the attention scores to not be too large.
        # It incurs a penalty if any of them has an absolute value greater than 50.0.
        # this should be outside the normal range of the attention scores.  We use
        # this mechanism instead of, say, a limit on entropy, because once the entropy
        # gets very small gradients through the softmax can become very small, and
        # some mechanisms like that become ineffective.
        attn_weights = penalize_abs_values_gt(attn_weights, limit=50.0, penalty=1.0e-04)

        if attn_mask is not None:
            assert (
                attn_mask.shape == (batch, 1, src_len)
                or attn_mask.shape == (batch, tgt_len, src_len)
            ), attn_mask.shape
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1), float("-inf"))

        attn_weights = attn_weights.view(batch * num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if random.random() < 0.001:
            print_attn_entropy(attn_weights, self.name)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        # (batch * head, tgt_len, head_dim)
        attn_output = torch.bmm(attn_weights, v)
        assert attn_output.shape == (batch * num_heads, tgt_len, head_dim), attn_output.shape

        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(tgt_len, batch, num_heads * head_dim)

        # (batch, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels: The number of channels of conv layers.
        kernel_size: Kernerl size of conv layers.
        bias: Whether to use bias in conv layers.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0, kernel_size

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.LayerNorm(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = nn.SiLU()

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor of shape(seq_len, batch, channels).
            padding_mask: padding mask of shape (batch, seq_len).

        Returns:
            Tensor: Output tensor (seq_len, batch, channels).

        """
        seq_len, batch, _ = x.shape

        x = x.permute(1, 2, 0)  # (batch, channels, seq_len).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, seq_len)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, seq_len)

        # 1D Depthwise Conv
        if padding_mask is not None:
            assert padding_mask.shape == (batch, seq_len), padding_mask.shape
            x.masked_fill_(padding_mask.unsqueeze(1).expand_as(x), 0.0)
        x = self.depthwise_conv(x)
        # x is (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, seq_len)

        return x.permute(2, 0, 1)


def print_attn_entropy(attn_weights: Tensor, module_name: str):
    # attn_weights: (num_heads, batch, tgt_len, src_len)
    (num_heads, batch, tgt_len, src_len) = attn_weights.shape

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            attn_weights = attn_weights.to(torch.float32)
            attn_weights_entropy = (
                -((attn_weights + 1.0e-20).log() * attn_weights)
                .sum(dim=-1)
                .mean(dim=(1, 2))
            )
            logging.info(
                f"name={module_name}, attn_weights_entropy = {attn_weights_entropy}"
            )


def _test_conformer(use_memory: bool = False):
    embed_dim = 512
    batch_size = 5
    seq_len = 100

    x = torch.randn(batch_size, seq_len, embed_dim)
    x_lens = torch.full((batch_size,), seq_len)

    if use_memory:
        memory_dim = 256
        memory = torch.randn(batch_size, 10, memory_dim)
        memory_lens = torch.full((batch_size,), 10)
    else:
        memory_dim = None
        memory = None
        memory_lens = None

    m = Conformer(embed_dim=embed_dim, memory_dim=memory_dim)
    y = m(x, x_lens, memory, memory_lens)

    assert y.shape == (batch_size, seq_len, embed_dim), y.shape
    print(f"use_memory={use_memory}, passed")


if __name__ == "__main__":
    _test_conformer(use_memory=False)
    _test_conformer(use_memory=True)
