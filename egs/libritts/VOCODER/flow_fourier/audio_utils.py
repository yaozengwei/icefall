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


from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class InverseMelScale(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ):
        super().__init__()
        # Compute the inverse filter banks using the pseudo inverse
        f_max = f_max or float(sample_rate // 2)
        fb = F.melscale_fbanks(
            (n_fft // 2 + 1), f_min, f_max, n_mels, sample_rate, norm, mel_scale
        )  # (F, n_mels)
        # Using pseudo-inverse is faster than calculating the least-squares in each
        # forward pass and experiments show that they converge to the same solution
        self.register_buffer("fb", torch.linalg.pinv(fb).transpose(0, 1))  # (F, n_mels)

    def forward(self, melspec: torch.Tensor) -> torch.Tensor:
        # Flatten the melspec except for the frequency and time dimension
        shape = melspec.shape
        melspec = melspec.view(-1, shape[-2], shape[-1])

        fb = self.fb.unsqueeze(0)  # (1, F, n_mels)

        # Sythesize the stft specgram using the filter banks
        specgram = torch.matmul(fb, melspec)  # (*, F, time)

        specgram = specgram.abs()  # negative is non-physical; zero undesirable.

        # Unflatten the specgram (*, freq, time)
        specgram = specgram.view(shape[:-2] + (fb.shape[1], shape[-1]))

        return specgram


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: str = "hann_window",
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.onesided = onesided
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, audio: torch.Tensor, audio_lens: torch.Tensor):
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
            onesided=self.onesided,
        )
        spec_lens = 1 + torch.div(audio_lens, self.hop_length, rounding_mode="floor")
        assert spec.shape[2] == spec_lens.max().item()
        return spec, spec_lens


class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: str = "hann_window",
        onesided: bool = True,
        return_complex: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.onesided = onesided
        self.return_complex = return_complex
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor):
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )
        return audio
