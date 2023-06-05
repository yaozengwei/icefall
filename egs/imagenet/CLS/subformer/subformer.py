#!/usr/bin/env python3
# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey,
#                                                       Zengwei Yao)
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
import warnings
from typing import List, Optional, Tuple, Union
import logging
import torch
import random

# from encoder_interface import EncoderInterface
from scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    clip_grad,
    convert_num_channels,
)
from torch import Tensor, nn


try:
    import os
    import sys

    kernel_path = os.path.abspath(os.path.join(".."))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import (
        WindowProcess,
        WindowProcessReverse,
    )

except:  # noqa
    WindowProcess = None
    WindowProcessReverse = None
    print(
        "[Warning] Fused window process have not been installed. Please refer to get_started.md for installation."
    )


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))


class FeedforwardModule(nn.Module):
    """Feedforward module in Subformer model."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works done by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish invidual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Temporary embedding dimension used inside this module
         pos_dim: Smaller positional-encoding dim used after a projecction.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
        pos_dim: dimension at the output of this module.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        dropout_rate: FloatLike,
        height: int,
        width: int,
        length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.pos_dim = pos_dim
        self.dropout = Dropout2(dropout_rate)
        assert length_factor >= 1.0
        self.length_factor = length_factor
        self.height = height
        self.width = width

        self.pos_emb = None

        # linear transformation for positional encoding.
        self.linear_pos_row = ScaledLinear(
            embed_dim, pos_dim, bias=False, initial_scale=0.05
        )
        self.linear_pos_col = ScaledLinear(
            embed_dim, pos_dim, bias=False, initial_scale=0.05
        )

    def generate_pe(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Generate the positional encodings with given length."""
        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = (
            torch.arange(-(length - 1), length, device=device)
            .to(torch.float32)
            .unsqueeze(1)
        )

        freqs = 1 + torch.arange(self.embed_dim // 2, device=device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = (
            compression_length
            * x.sign()
            * ((x.abs() + compression_length).log() - math.log(compression_length))
        )

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan()  # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.
        # pe: (2 * length - 1, embed_dim)
        return pe

    def generate_2d_pe(self, x: Tensor) -> Tensor:
        if self.pos_emb is not None:
            # x: (B, C, W, H)
            # self.pos_emb: (W*H, W*H, pos_dim)
            assert self.pos_emb.size(0) == self.pos_emb.size(1) == x.size(2) * x.size(3)
            # Note: TorchScript doesn't implement operator== for torch.Device
            if self.pos_emb.dtype != x.dtype or str(self.pos_emb.device) != str(
                x.device
            ):
                self.pos_emb = self.pos_emb.to(dtype=x.dtype, device=x.device)
            return self.pos_emb

        H = self.height
        W = self.width

        row = self.generate_pe(length=H, device=x.device, dtype=x.dtype)
        col = self.generate_pe(length=W, device=x.device, dtype=x.dtype)

        # (1, 2 * H - 1, pos_dim)
        row = self.linear_pos_row(self.dropout(row.unsqueeze(0)))
        # (1, 2 * W - 1, pos_dim)
        col = self.linear_pos_col(self.dropout(col.unsqueeze(0)))

        # expand the '1' dimension to seq_len; this introduces a dimension that
        # 'does nothing', just creates copies, as a workaround for lack of torch support
        # for negative strides.
        row = row.expand(H, 2 * H - 1, self.pos_dim).contiguous()
        (useless_stride, seq_stride, channel_stride) = row.stride()
        row = row.as_strided(
            (H, H, self.pos_dim),
            (useless_stride - seq_stride, seq_stride, channel_stride),
            storage_offset=seq_stride * (H - 1),
        )  # (H, H, pos_dim)

        # expand the '1' dimension to seq_len; this introduces a dimension that
        # 'does nothing', just creates copies, as a workaround for lack of torch support
        # for negative strides.
        col = col.expand(W, 2 * W - 1, self.pos_dim).contiguous()
        (useless_stride, seq_stride, channel_stride) = col.stride()
        col = col.as_strided(
            (W, W, self.pos_dim),
            (useless_stride - seq_stride, seq_stride, channel_stride),
            storage_offset=seq_stride * (W - 1),
        )  # (W, W, pos_dim)

        row = row[:, None, :, None, :].expand(H, W, H, W, -1)
        col = col[None, :, None, :, :].expand(H, W, H, W, -1)
        pos_emb = torch.stack([row, col], dim=-1)
        pos_emb = pos_emb.view(H * W, H * W, -1)

        self.pos_emb = pos_emb
        return self.pos_emb

    def forward(self, x: torch.Tensor) -> Tensor:
        """Create positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch_size, height, width, num_channels_in)

        Returns:
            positional embedding, of shape (batch_size, height*width, heigh*width, pos_dim*2).
        """
        pos_emb = self.generate_2d_pe(x)
        batch_size = x.size(0)

        length = self.height * self.width
        pos_emb = pos_emb.unsqueeze(0).expand(
            batch_size, length, length, self.pos_dim * 2
        )

        return pos_emb  # (batch_size, W*H, W*H, pos_dim)


def _test_rel_pos_enc():
    pos_enc = CompactRelPositionalEncoding(
        embed_dim=384, pos_dim=4, dropout_rate=0.1, height=7, width=6
    )
    x = torch.randn(2, 384, 7, 6)
    pos_emb = pos_enc(x)
    pos_emb = pos_enc(x)
    print(pos_emb.shape)

    pos_emb = pos_emb.reshape(2, 7, 6, 7, 6, 8)
    # a simple test
    assert torch.equal(pos_emb[:, 1, 2, 3, 4, :], pos_emb[:, 2, 3, 4, 5, :])
    assert torch.equal(pos_emb[:, 4, 3, 2, 1, :], pos_emb[:, 5, 4, 3, 2, :])


if __name__ == "__main__":
    _test_rel_pos_enc()
