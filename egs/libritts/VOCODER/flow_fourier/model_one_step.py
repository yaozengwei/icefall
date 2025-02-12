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
import random
import logging
from typing import Optional, Tuple

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
from convnext import AudioConvNeXt, MelEncoder


class OneStepVocoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        sampling_rate: int = 24000,
        n_ffts: Tuple[int] = (512, 256, 128, 64),
        hop_lengths: Tuple[int] = (256, 128, 64, 32),
        mel_n_fft: int = 512,
        mel_hop_length: int = 256,
        mel_enc_channels: int = 512,
        mel_enc_num_layers: int = 4,
        convnext_num_layers: int = (8, 8, 8, 8),
        convnext_channels: int = (768, 384, 192, 96),
        from_inv_mel: bool = True,
        init_noise_scale: float = 0.1,
    ):
        super().__init__()
        self.num_branches = len(n_ffts)
        assert len(hop_lengths) == self.num_branches
        assert len(convnext_num_layers) == self.num_branches
        assert len(convnext_channels) == self.num_branches
        # These two arguments should be consistent with the model used to get the inverted noise
        self.from_inv_mel = from_inv_mel
        self.init_noise_scale = init_noise_scale  # emprical, need to tune

        self.mel_encoder = MelEncoder(
            n_mels=n_mels,
            channels=mel_enc_channels,
            num_layers=mel_enc_num_layers,
        )

        self.estimators = nn.ModuleList([
            AudioConvNeXt(
                n_fft=n_ffts[i],
                hop_length=hop_lengths[i],
                cond_channels=mel_enc_channels,
                mel_hop_length=mel_hop_length,
                convnext_channels=convnext_channels[i],
                convnext_num_layers=convnext_num_layers[i],
                num_outputs=1,
                use_t=False,
                use_dest_t=False,
                analytic=False,
            )
            for i in range(self.num_branches)
        ])

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
            n_mels=n_mels,
            sample_rate=sampling_rate,
            n_fft=mel_n_fft,
        )

        self.ifft = ISTFT(n_fft=mel_n_fft, hop_length=mel_hop_length)

        self.apply(self._init_weights)

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
        return est_audio * 2.0
        # return est_audio

    def process_model(
        self,
        x: Tensor,
        audio_lens: Tensor,
        cond: Tensor,
        branch_drop_rate: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        branch_outputs = torch.stack([
            estimator(audio=x, audio_lens=audio_lens, mel=cond)
            for estimator in self.estimators
        ], dim=1)  # (batch, num_branches, 1, time)

        # fuse all branches
        if not self.training or branch_drop_rate <= 0:
            output = branch_outputs.mean(dim=1)  # (batch, 1, time)

        else:
            if random.random() < 0.05:
                logging.info(f"branch_drop_rate={branch_drop_rate}")
            # At the start of training, apply random branch masking
            mask = torch.rand(branch_outputs.shape[:2], device=x.device) > branch_drop_rate
            mask[mask.sum(dim=1) == 0] = True  # Unmask if all branches are dropped
            weight = (mask / mask.sum(dim=1, keepdim=True))[:, :, None, None]  # (batch, num_branches, 1, 1)
            output = (branch_outputs * weight).sum(dim=1)  # (batch, 1, time)

        return output[:, 0]

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        inv_noise: torch.Tensor,
        mel_scaling_loss: bool = True,
        branch_drop_rate: float = 0.0,
        eps: float = 1e-8,
    ) -> Tuple[Tensor]:
        x1 = audio
        x0 = inv_noise

        # input condition
        mel_spec = self.mel(audio)  # (batch, n_mels, time)
        cond = self.mel_encoder(mel_spec.sqrt())

        x1_pred = self.process_model(
            x=x0,
            audio_lens=audio_lens,
            cond=cond,
            branch_drop_rate=branch_drop_rate,
        )
        # shape of vt, x_mid, x_dest should be: (batch_size, time)

        # compute losses
        main_loss = self.compute_loss(
            pred=x1_pred,
            target=x1,
            audio_lens=audio_lens,
            mel_scaling_loss=mel_scaling_loss,
            mel_spec=mel_spec,
        )

        loss = (main_loss,)
        return loss

    def compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        audio_lens: Tensor,
        mel_scaling_loss: bool = True,
        mel_spec: Optional[Tensor] = None,
        loss_power: float = 0.5,
        eps: float = 1.0e-07,
    ) -> Tensor:
        err = pred - target

        if not mel_scaling_loss:
            pad_mask = make_pad_mask(audio_lens).logical_not()  # (batch, time)
            loss = ((err ** 2) * pad_mask).sum() / pad_mask.sum()
        else:
            assert mel_spec is not None
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
            loss = (loss * pad_mask).sum() / (pad_mask.sum() * err_mel_spec.shape[1])

        return loss

    def infer(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        inv_noise: Optional[torch.Tensor] = None,
        clamp_pred: bool = True,
        log_mel_diff: bool = False,
    ) -> Tensor:
        mel_spec = self.mel(audio)  # (batch, n_mels, time)

        if inv_noise is None:
            # sample noise p(x_0)
            if self.from_inv_mel:
                noise = self.reconstuct_audio_with_random_phase(mel_spec)
                noise = convert_length(noise, audio.shape[-1])
            else:
                # scale x0 by x1's std in training
                noise = torch.randn_like(audio) * self.init_noise_scale
        else:
            noise = inv_noise

        cond = self.mel_encoder(mel_spec.sqrt())
        pred_audio = self.process_model(x=noise, audio_lens=audio_lens, cond=cond)

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
