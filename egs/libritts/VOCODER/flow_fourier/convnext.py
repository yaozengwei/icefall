#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (authors: Daniel Povey, Zengwei Yao)
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


from typing import Optional

import math
import torch
from audio_utils import STFT, ISTFT
from icefall.utils import make_pad_mask
from torch import nn
from torch import Tensor


def convert_length(x: Tensor, length: int) -> Tensor:
    # return x with the last dimension either truncated or extended with zeros,
    # to 'length'.
    if length <= x.shape[-1]:
        return x[..., :length]
    else:
        shape = list(x.shape)
        shape[-1] = length - shape[-1]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat((x, zeros), dim=-1)


def fft_to_real(fft: Tensor):
    """
    fft: (batch_size, fft_channels, fft_frames), complex.
    Returns: real_fft: (batch_size, 2 * fft_channels, fft_frames), real
    """
    (batch_size, _, fft_frames) = fft.shape
    real_fft = torch.view_as_real(fft).permute(0, 3, 1, 2).reshape(batch_size, -1, fft_frames)
    return real_fft


def real_to_fft(real_fft: Tensor):
    """
    real_fft: (batch_size, 2 * fft_channels, fft_frames), real
    Returns: fft: (batch_size, fft_channels, fft_frames), complex.
    """
    (batch_size, _, fft_frames) = real_fft.shape
    real_fft = real_fft.reshape(batch_size, 2, -1, fft_frames).permute(0, 2, 3, 1)
    fft = torch.view_as_complex(real_fft.contiguous())
    return fft


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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        cond_channels: Optional[int] = None,
        time_embed_channels: Optional[int] = None,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.pwconv1 = nn.Linear(channels, hidden_channels, bias=False)
        self.act = nn.LeakyReLU()
        self.pwconv2 = nn.Linear(hidden_channels, channels, bias=False)

        if cond_channels is not None:
            self.cond_proj = nn.Linear(cond_channels, channels, bias=False)
        if time_embed_channels is not None:
            self.time_embed_proj = nn.Linear(time_embed_channels, channels)

        self.residual_scale = nn.Parameter(torch.full((1,), residual_scale))

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        time_embed: Optional[torch.Tensor] = None,
        length_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, in_channels, time)
            cond: (batch_size, cond_channels, time)
            time_embed: (batch, channels)
            length_mask: (batch_size, 1, time)
        """
        residual = x

        if length_mask is not None:
            x = x * length_mask
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # Add condition and time embeddings
        if cond is not None:
            assert hasattr(self, "cond_proj")
            cond = self.cond_proj(cond.transpose(1, 2))
            x = x + cond
        if time_embed is not None:
            assert hasattr(self, "time_embed_proj")
            time_embed = self.time_embed_proj(time_embed).unsqueeze(1)
            x = x * (1. + time_embed)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = x + residual * self.residual_scale

        return x


class MelEncoder(nn.Module):
    """ConvNeXt-based mel-spectrogram encoder."""
    def __init__(self, n_mels: int = 80, channels: int = 512, num_layers: int = 4):
        super().__init__()
        self.in_proj = nn.Conv1d(n_mels, channels, 1, bias=False)
        self.convnext_blocks = nn.ModuleList(
            [
                ConvNeXtBlock(channels=channels, hidden_channels=channels * 3)
                for _ in range(num_layers)
            ]
        )
        self.act = nn.LeakyReLU()

    def forward(
        self,
        x: torch.Tensor,
        length_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_mels, time)
            length_mask: (batch, 1, time)
        """
        x = self.in_proj(x)
        for block in self.convnext_blocks:
            x = block(x, length_mask=length_mask)
        x = self.act(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt model that processes the Fourier spectral coefficients."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        channels: int,
        num_layers: int,
        use_dest_t: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_proj = nn.Conv1d(in_channels, channels, 1, bias=False)

        self.time_embed = SinusoidalPosEmb(channels)
        time_embed_hidden = channels * 3
        self.time_mlp = nn.Sequential(
            nn.Linear(channels if not use_dest_t else channels * 2, time_embed_hidden),
            nn.SiLU(),
            nn.Linear(time_embed_hidden, channels),
        )

        cond_embed_hidden = channels * 3
        self.cond_mlp = nn.Sequential(
            nn.Conv1d(cond_channels, cond_embed_hidden, kernel_size=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv1d(cond_embed_hidden, channels, kernel_size=1, bias=False),
        )

        self.convnext_blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    channels=channels,
                    hidden_channels=channels * 3,
                    cond_channels=channels,
                    time_embed_channels=channels,
                    residual_scale=0.9,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Conv1d(channels, out_channels, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        dest_t: Optional[torch.Tensor] = None,
        length_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time)
            cond: (batch, cond_channels, time)
            t: (batch,)
            dest_t: (batch,)
            length_mask: (batch, 1, time)

        Returns:
            x: (batch, out_channels, time)
        """
        x = self.in_proj(x)

        if dest_t is not None:
            time_embed = torch.cat([self.time_embed(t), self.time_embed(dest_t)], dim=2)
        else:
            time_embed = self.time_embed(t)
        time_embed = self.time_mlp(time_embed)  # (batch, channels)

        cond = self.cond_mlp(cond)

        for block in self.convnext_blocks:
            x = block(x, cond=cond, time_embed=time_embed, length_mask=length_mask)

        x = self.out_proj(x)

        return x


class AudioConvNeXt(nn.Module):
    """ConvNeXt-based model that processes audio wavforms"""
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        cond_channels: int,
        mel_hop_length: int,
        convnext_channels: int,
        convnext_num_layers: int,
        num_outputs: int = 1,
        use_dest_t: bool = False,
        analytic: bool = False,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.fft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=not analytic,
        )

        # mel_hop_length should be integer multiple of hop_length.
        assert mel_hop_length % hop_length == 0, (mel_hop_length, hop_length)
        self.mel_upsample_factor = mel_hop_length // hop_length

        real_fft_channels = n_fft + 2 if not analytic else 2 * n_fft
        self.convnext = ConvNeXt(
            in_channels=real_fft_channels,
            out_channels=real_fft_channels * num_outputs,
            cond_channels=cond_channels,
            channels=convnext_channels,
            num_layers=convnext_num_layers,
            use_dest_t=use_dest_t,
        )

        self.ifft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            onesided=not analytic,
            return_complex=analytic,
        )

    def forward(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        t: Tensor,
        mel: Tensor,
        dest_t: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """
        Args:
            audios: (batch_size, audio_len)
            audio_lens: (batch_size,)
            t: (batch_size,)
            mel: (batch, n_mels, mel_frames)
            dest_t: (batch_size,)

        Returns: (batch_size, audio_len)
        """
        fft, fft_lens = self.fft(audios, audio_lens)
        # fft: (batch, fft_channels, fft_frames); complex.
        fft_real = fft_to_real(fft)
        # fft_real: (batch, 2 * fft_channels, fft_frames)
        mel = self.upsample_mel(mel, fft.shape[2])
        # now mel: (batch, n_mel, fft_frames)

        fft_length_mask = make_pad_mask(fft_lens, max_len=fft.shape[2]).logical_not().unsqueeze(1)
        fft_real = self.convnext(
            fft_real, cond=mel, t=t, dest_t=dest_t, length_mask=fft_length_mask
        )

        batch_size, channels, fft_frames = fft_real.shape
        num_outputs = self.num_outputs
        fft_real = fft_real.reshape(batch_size * num_outputs, channels // num_outputs, fft_frames)

        fft = real_to_fft(fft_real)
        audios = self.ifft(fft)
        audios = audios.reshape(batch_size, num_outputs, audios.shape[-1])
        audios = convert_length(audios, audio_lens.max())

        audio_length_mask = make_pad_mask(audio_lens, max_len=audios.shape[-1]).logical_not()
        audios = audios * audio_length_mask.unsqueeze(1)

        return audios

    def upsample_mel(self, mel: Tensor, fft_frames: int) -> Tensor:
        """Upsample mel coefficients, if necessary, to match the FFT coefficients.
        Args:
            Mel: (batch_size, n_mels, mel_frames)
        """
        f = self.mel_upsample_factor
        if f != 1:
            (batch_size, n_mels, mel_frames) = mel.shape
            mel = mel.unsqueeze(-1).expand(batch_size, n_mels, mel_frames, f)
            mel = mel.reshape(batch_size, n_mels, -1)
        mel = convert_length(mel, fft_frames)
        return mel
