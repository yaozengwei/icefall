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


import math
import logging
from typing import Optional, Tuple, Union

import torch
from torchaudio.transforms import MelSpectrogram
from torch import Tensor, nn
from icefall.utils import make_pad_mask
from audio_utils import (
    ISTFT,
    InverseMelScale,
    convert_length,
    safe_log,
)
from transformer import AudioLocalGlogalTransformer, MelEncoder


class Vocoder(nn.Module):
    def __init__(
        self,
        sampling_rate: int = 24000,
        n_mels: int = 80,
        mel_n_fft: int = 512,
        mel_hop_length: int = 256,
        mel_enc_channels: int = 512,
        mel_enc_num_layers: int = 4,
        mel_enc_hidden_factor: int = 4,
        kernel_size: int = 16,
        stride: int = 8,
        embed_dim: int = 256,
        chunk_size: int = 160,
        num_encoders: int = 2,
        local_num_layers: int = 8,
        global_num_layers: int = 8,
        num_heads: int = 8,
        ff_hidden_factor: int = 4,
        dropout: float = 0.0,
        use_res_scale: bool = True,
        use_pos_enc: bool = True,
        use_norm: bool = True,
        use_skip: bool = True,
        use_skip_scale: bool = True,
        from_inv_mel: bool = True,
        init_noise_scale: float = 0.1,
    ):
        super().__init__()
        self.from_inv_mel = from_inv_mel
        self.init_noise_scale = init_noise_scale  # emprical, need to tune

        self.mel_encoder = MelEncoder(
            n_mels=n_mels,
            channels=mel_enc_channels,
            num_layers=mel_enc_num_layers,
            hidden_factor=mel_enc_hidden_factor,
            use_res_scale=use_res_scale,
        )

        self.estimator = AudioLocalGlogalTransformer(
            kernel_size=kernel_size,
            stride=stride,
            embed_dim=embed_dim,
            cond_dim=mel_enc_channels,
            out_dim=embed_dim,
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

        self.mel = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=mel_n_fft,
            win_length=mel_n_fft,
            hop_length=mel_hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            center=True,
            power=2,
        )

        self.inv_mel = InverseMelScale(
            n_mels=n_mels, sample_rate=sampling_rate, n_fft=mel_n_fft
        )

        self.ifft = ISTFT(n_fft=mel_n_fft, hop_length=mel_hop_length, center=True)

        # self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.015)
            if hasattr(m, 'bias') and isinstance(m.bias, Tensor):
                nn.init.constant_(m.bias, 0)

    def reconstuct_audio_with_random_phase(self, mel_spec: Tensor) -> Tensor:
        est_mag = self.inv_mel(mel_spec) ** 0.5   # mel -> power magnitude -> amplitude
        # Get a random phase, (not a strict form)
        pi = math.pi
        random_phase = torch.rand_like(est_mag) * (2 * pi) - pi
        est_fft = est_mag * torch.exp(random_phase * 1j)
        est_audio = self.ifft(est_fft)
        # return est_audio * 2.0
        return est_audio

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        mel_scaling_loss: bool = True,
    ) -> Tuple[Tensor]:
        x1 = audio

        # input condition
        mel_spec = self.mel(audio)  # (batch, n_mels, time)

        # sample noise p(x_0)
        if self.from_inv_mel:
            x0 = self.reconstuct_audio_with_random_phase(mel_spec)
            x0 = convert_length(x0, audio.shape[-1])
        else:
            # scale x0 by x1's std in training
            x0 = torch.randn_like(x1) * self.init_noise_scale

        t = torch.rand((audio.shape[0], 1), device=audio.device, dtype=audio.dtype)
        xt = (1.0 - t) * x0 + t * x1
        ut = x1 - x0

        cond = self.mel_encoder(mel_spec.sqrt())
        vt = self.estimator(audio=xt, audio_lens=audio_lens, cond=cond, t=t.flatten())

        loss = self.compute_loss(
            pred=vt,
            target=ut,
            audio_lens=audio_lens,
            mel_scaling_loss=mel_scaling_loss,
            mel_spec=mel_spec,
        )

        return loss

    def compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        audio_lens: Tensor,
        loss_scale: Union[float, Tensor] = 1.0,
        mel_scaling_loss: bool = False,
        mel_spec: Optional[Tensor] = None,
        loss_power: float = 0.5,
        eps: float = 1.0e-07,
    ) -> Tensor:
        if mel_scaling_loss:
            assert mel_spec is not None

        err = pred - target

        if not mel_scaling_loss:
            pad_mask = make_pad_mask(audio_lens).logical_not()  # (batch, time)
            loss = ((err ** 2) * loss_scale * pad_mask).sum() / pad_mask.sum()
        else:
            if isinstance(loss_scale, Tensor):
                loss_scale = loss_scale.unsqueeze(1)  # (batch, 1) -> (batch, 1, 1)

            err_mel_spec = self.mel(err)
            # the err_mel_spec.sum() is just an aggregation of squared errors for FFT bins, with
            # frequency-specific weightings.  Scaling by (mel_spec + eps) ** -loss_power is a heuristic
            # scale that puts more weight on quieter regions of the spectrum, where presumably
            # differences would be more audible; the choice of 0.5 for loss_power is arbitrary, it
            # could be anywhere between 0 (no correction for volume) and 1 (fully invariant to
            # local volume).
            mel_spec_lens = 1 + torch.div(audio_lens, self.mel.hop_length, rounding_mode="floor")
            assert err_mel_spec.shape[2] == mel_spec_lens.max().item()
            pad_mask = make_pad_mask(mel_spec_lens).logical_not().unsqueeze(1)  # (batch, 1, time)
            loss = err_mel_spec * ((mel_spec + eps) ** -loss_power)
            loss = (loss * loss_scale * pad_mask).sum() / (pad_mask.sum() * err_mel_spec.shape[1])

        return loss

    def infer(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        n_timesteps: int = 8,
        clamp_pred: bool = True,
        log_mel_diff: bool = False,
    ) -> Tensor:
        mel_spec = self.mel(audio)  # (batch, n_mels, time)

        # sample noise p(x_0)
        if self.from_inv_mel:
            noise = self.reconstuct_audio_with_random_phase(mel_spec)
            noise = convert_length(noise, audio.shape[-1])
        else:
            # scale x0 by x1's std in training
            noise = torch.randn_like(audio) * self.init_noise_scale

        cond = self.mel_encoder(mel_spec.sqrt())

        # use fixed euler solver for ODEs.
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=noise.device)
        t, dt = t_span[0], t_span[1] - t_span[0]
        x = noise
        batch_size = x.shape[0]
        for step in range(1, len(t_span)):
            vt = self.estimator(
                audio=x,
                audio_lens=audio_lens,
                cond=cond,
                t=t[None].expand(batch_size),
            )
            x = x + vt * dt
            t = t_span[step]

        pred_audio = x
        if clamp_pred:
            pred_audio = pred_audio.clamp(min=-1.0, max=1.0)

        if log_mel_diff:
            mel_spec_noise = self.mel(noise)  # (batch, n_mels, time)
            mel_spec_infer = self.mel(pred_audio)  # (batch, n_mels, time)

            def mel_diff(x, y):
                return (safe_log(x) - safe_log(y)).abs().mean().item()

            logging.info(
                f"Mel-diffs ref-noise={mel_diff(mel_spec, mel_spec_noise)}, "
                f"ref-infer={mel_diff(mel_spec, mel_spec_infer)}, "
                f"noise-infer={mel_diff(mel_spec_noise, mel_spec_infer)}"
            )

            def mel_diff2(x, y):  # red
                return (safe_log(x) - safe_log(y)).mean().item()

            logging.info(
                f"Mel-diffs-noabs ref-noise={mel_diff2(mel_spec, mel_spec_noise)}, "
                f"ref-infer={mel_diff2(mel_spec, mel_spec_infer)}, "
                f"noise-infer={mel_diff2(mel_spec_noise, mel_spec_infer)}"
            )

        return pred_audio
