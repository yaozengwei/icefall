#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (authors: Zengwei Yao)
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


import random
from typing import Optional

import math
import torch
from audio_utils import convert_length
from icefall.utils import make_pad_mask
from torch import nn
from torch import Tensor


# from https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/transformer/Transformer.py
class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(
                f"Cannot use sin/cos positional encoding with odd channels (got channels={input_size})"
            )
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor
            Input feature shape (batch, time, fea)

        Returns
        -------
        The positional encoding.
        """
        return self.pe[:, : x.size(1)].clone().detach()


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# From https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/scaling.py
class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_factor: int = 4,
        dropout: float = 0.0,
        use_res_scale: bool = True,
    ):
        super().__init__()
        self.use_res_scale = use_res_scale

        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.0
        )

        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        ff_hidden_dim = embed_dim * ff_hidden_factor
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        # TODO: test removing nonlin activations here
        self.cond_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        )
        self.time_embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim)
        )

        if use_res_scale:
            self.res_scale1 = nn.Parameter(torch.full((embed_dim,), 0.9))
            self.res_scale2 = nn.Parameter(torch.full((embed_dim,), 0.9))

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        time_embed: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (time, batch_size, embed_dim)
            cond: (time, batch_size, embed_dim)
            time_embed: (batch_size, embed_dim)
            attn_mask: (L, S), where L is the target length, S is the source length
            key_padding_mask: (batch_size, S), where S is the source length

        Returns:
            x: (time, batch_size, embed_dim)
        """
        # each of shape (time, batch_size, embed_dim)
        shift_attn, shift_ff = self.cond_proj(cond).chunk(2, dim=-1)
        # each of shape (batch_size, embed_dim)
        scale_attn, scale_ff = self.time_embed_proj(time_embed).chunk(2, dim=-1)

        attn_in = self.norm1(x) * (1.0 + scale_attn.unsqueeze(0)) + shift_attn
        attn_out = self.attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        if self.use_res_scale:
            res_scale1 = limit_param_value(
                self.res_scale1, min=0.5, max=1.0, training=self.training
            )
            x = attn_out + x * res_scale1
        else:
            x = attn_out + x

        ff_in = self.norm2(x) * (1.0 + scale_ff.unsqueeze(0)) + shift_ff
        ff_out = self.ff(ff_in)
        if self.use_res_scale:
            res_scale2 = limit_param_value(
                self.res_scale2, min=0.5, max=1.0, training=self.training
            )
            x = ff_out + x * res_scale2
        else:
            x = ff_out + x

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_factor: int = 4,
        dropout: float = 0.0,
        use_res_scale: bool = True,
        use_pos_enc: bool = True,
        use_norm: bool = True,
        use_skip: bool = True,
        use_skip_scale: bool = True,
    ):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.use_norm = use_norm
        self.use_skip = use_skip
        self.use_skip_scale = use_skip_scale

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_factor=ff_hidden_factor,
                dropout=dropout,
                use_res_scale=use_res_scale,
            )
            for _ in range(num_layers)
        ])

        if use_pos_enc:
            self.pos_enc = PositionalEncoding(input_size=embed_dim, max_len=100000)

        if use_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        if use_skip and use_skip_scale:
            self.skip_scale = nn.Parameter(torch.full((embed_dim,), 0.9))

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        time_embed: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (time, batch_size, embed_dim)
            cond: (time, batch_size, embed_dim)
            time_embed: (batch_size, embed_dim)
            attn_mask: (L, S), where L is the target length, S is the source length
            key_padding_mask: (batch_size, S), where S is the source length

        Returns:
            x: (time, batch_size, embed_dim)
        """
        x_skip = x
        if self.use_pos_enc:
            # shape: (1, time, embed_dim)
            pos_enc = self.pos_enc(x.transpose(0, 1)).transpose(0, 1)
            x = x + pos_enc

        for layer in self.layers:
            x = layer(
                x=x,
                cond=cond,
                time_embed=time_embed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        if self.use_norm:
            x = self.norm(x)

        if self.use_skip:
            if self.use_skip_scale:
                skip_scale = limit_param_value(
                    self.skip_scale, min=0.2, max=1.0, training=self.training
                )
                x = x + x_skip * skip_scale
            else:
                x = x + x_skip

        return x


class LocalGlogalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        chunk_size: int,
        num_encoders: int,
        local_num_layers: int,
        global_num_layers: int,
        num_heads: int,
        ff_hidden_factor: int = 4,
        dropout: float = 0.0,
        use_res_scale: bool = True,
        use_pos_enc: bool = True,
        use_norm: bool = True,
        use_skip: bool = True,
        use_skip_scale: bool = True,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_encoders = num_encoders

        self.local_encoders = nn.ModuleList([
            TransformerEncoder(
                num_layers=local_num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_factor=ff_hidden_factor,
                dropout=dropout,
                use_res_scale=use_res_scale,
                use_pos_enc=use_pos_enc,
                use_norm=use_norm,
                use_skip=use_skip,
                use_skip_scale=use_skip_scale,
            )
            for _ in range(num_encoders)
        ])
        self.global_encoders = nn.ModuleList([
            TransformerEncoder(
                num_layers=local_num_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_factor=ff_hidden_factor,
                dropout=dropout,
                use_res_scale=use_res_scale,
                use_pos_enc=use_pos_enc,
                use_norm=use_norm,
                use_skip=use_skip,
                use_skip_scale=use_skip_scale,
            )
            for _ in range(num_encoders - 1)
        ])
        self.output = nn.Sequential(
            nn.LeakyReLU(), nn.Conv1d(embed_dim, out_dim, kernel_size=1, bias=False)
        )

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        cond: Tensor,
        time_embed: Tensor,
    ) -> Tensor:
        """
        Args:
            x: (batch_size, embed_dim, time)
            x_lens: (batch_size,)
            cond: (batch_size, embed_dim, time)
            time_embed: (batch_size, embed_dim)

        Returns:
            x: (batch_size, embed_dim, time)
        """
        batch_size, embed_dim, time = x.shape
        chunk_size = self.chunk_size

        pad_mask = make_pad_mask(x_lens)  # (batch_size, time)

        num_chunks = (time + chunk_size - 1) // chunk_size
        pad_len = num_chunks * chunk_size - time
        if pad_len > 0:
            x = nn.functional.pad(x, (0, pad_len), value=0.0)
            pad_mask = nn.functional.pad(pad_mask, (0, pad_len), value=True)
            cond = nn.functional.pad(cond, (0, pad_len), value=0.0)

        x = x.view(batch_size, embed_dim, num_chunks, chunk_size)
        x = x.permute(3, 0, 2, 1).reshape(chunk_size, batch_size * num_chunks, embed_dim)

        local_pad_mask = pad_mask.view(batch_size, num_chunks, chunk_size)
        local_pad_mask = local_pad_mask.reshape(batch_size * num_chunks, chunk_size)

        local_cond = cond.view(batch_size, embed_dim, num_chunks, chunk_size)
        local_cond = local_cond.permute(3, 0, 2, 1)
        local_cond = local_cond.reshape(chunk_size, batch_size * num_chunks, embed_dim)

        local_time_embed = time_embed.unsqueeze(1).expand(batch_size, num_chunks, embed_dim)
        local_time_embed = local_time_embed.reshape(batch_size * num_chunks, embed_dim)

        global_pad_mask = pad_mask[:, ::chunk_size]  # (batch_size, num_chunks)

        global_cond = local_cond.mean(0).reshape(batch_size, num_chunks, embed_dim)
        global_cond = global_cond.permute(1, 0, 2)  # (num_chunks, batch_size, embed_dim)

        for i in range(self.num_encoders):
            x = self.local_encoders[i](
                x=x,
                cond=local_cond,
                time_embed=local_time_embed,
                key_padding_mask=local_pad_mask,
            )  # (chunk_size, batch_size * num_chunks, embed_dim)
            if i < self.num_encoders - 1:
                mem = x.mean(dim=0).reshape(batch_size, num_chunks, embed_dim)
                mem = mem.permute(1, 0, 2)  # (num_chunks, batch_size, embed_dim)
                mem = self.global_encoders[i](
                    x=mem,
                    cond=global_cond,
                    time_embed=time_embed,
                    key_padding_mask=global_pad_mask,
                )
                mem = mem.permute(1, 0, 2).reshape(batch_size * num_chunks, embed_dim)
                x = x + mem.unsqueeze(0)

        x = x.view(chunk_size, batch_size, num_chunks, embed_dim).permute(1, 3, 2, 0)
        x = x.reshape(batch_size, embed_dim, num_chunks * chunk_size)

        x = self.output(x) * pad_mask.logical_not().unsqueeze(1)
        x = x[:, :, :time]  # (batch_size, embed_dim, time)

        return x


class AudioLocalGlogalTransformer(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        embed_dim: int,
        cond_dim: int,
        out_dim: int,
        chunk_size: int,
        num_encoders: int,
        local_num_layers: int,
        global_num_layers: int,
        num_heads: int,
        ff_hidden_factor: int = 4,
        dropout: float = 0.0,
        use_res_scale: bool = True,
        use_pos_enc: bool = True,
        use_norm: bool = True,
        use_skip: bool = True,
        use_skip_scale: bool = True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            nn.LeakyReLU()
        )
        self.decoder = nn.ConvTranspose1d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

        self.net = LocalGlogalTransformer(
            embed_dim=embed_dim,
            out_dim=out_dim,
            chunk_size=chunk_size,
            num_encoders=num_encoders,
            local_num_layers=local_num_layers,
            global_num_layers=global_num_layers,
            num_heads=num_heads,
            ff_hidden_factor=ff_hidden_factor,
            dropout=dropout,
            use_res_scale=use_res_scale,
            use_pos_enc=use_pos_enc,
            use_norm=use_norm,
            use_skip=use_skip,
            use_skip_scale=use_skip_scale,
        )

        ff_hidden_dim = embed_dim * ff_hidden_factor
        self.time_embed = SinusoidalPosEmb(embed_dim)
        self.time_ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.SiLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.cond_ff = nn.Sequential(
            nn.Conv1d(cond_dim, ff_hidden_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(ff_hidden_dim, embed_dim, kernel_size=1, bias=False),
        )

    def upsample_cond(self, cond: Tensor, length: int) -> Tensor:
        """Upsample condition input to target length, if necessary.
        Args:
            cond: (batch_size, embed_dim, cond_len)
        """
        batch_size, embed_dim, cond_len = cond.shape
        assert cond_len <= length, (cond_len, length)
        up_factor = length // cond_len
        if up_factor != 1:
            cond = cond.unsqueeze(-1).expand(batch_size, embed_dim, cond_len, up_factor)
            cond = cond.reshape(batch_size, embed_dim, cond_len * up_factor)
        cond = convert_length(cond, length)
        return cond

    def forward(
        self,
        audio: Tensor,
        audio_lens: Tensor,
        cond: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Args:
            audio: (batch_size, time)
            audio_lens: (batch_size,)
            cond: (batch_size, cond_dim, cond_len)
            t: (batch_size,)

        Returns:
            x: (batch_size, time)
        """
        batch_size, time = audio.shape

        x = self.encoder(audio.unsqueeze(1))  # (batch_size, embed_dim, time2)
        x_lens = (audio_lens - self.kernel_size) // self.stride + 1
        assert x_lens.max().item() == x.shape[2]

        time_embed = self.time_embed(t)
        time_embed = self.time_ff(time_embed)  # (batch_size, embed_dim)

        cond = self.cond_ff(cond)  # (batch_size, embed_dim, cond_len)
        cond = self.upsample_cond(cond, length=x.shape[2])  # (batch, embed_dim, time2)

        x = self.net(x=x, x_lens=x_lens, cond=cond, time_embed=time_embed)

        x = self.decoder(x).squeeze(1)
        x = convert_length(x, time)  # (batch_size, time)

        return x









