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
from torch.nn import functional as F
from icefall.utils import make_pad_mask
from audio_utils import (
    ISTFT,
    InverseMelScale,
    STFT,
    convert_length,
    safe_log,
)
from convnext import AudioConvNeXt, MelEncoder


class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    def backward(ctx, x_grad):
        return x_grad * ctx.scale, None


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
        convnext_conv_kernel_sizes: int = (7, 7, 7, 7),
        from_inv_mel: bool = True,
        init_noise_scale: float = 0.1,
        use_disc_loss: bool = False,
        disc_mask_order: int = 3,
        use_fft_mag_loss: bool = False,
        use_log_mel_loss: bool = False,
        mag_power: int = 1,
    ):
        super().__init__()
        self.num_branches = len(n_ffts)
        assert len(hop_lengths) == self.num_branches
        assert len(convnext_num_layers) == self.num_branches
        assert len(convnext_channels) == self.num_branches
        assert len(convnext_conv_kernel_sizes) == self.num_branches
        # These two arguments should be consistent with the model used to get the inverted noise
        self.from_inv_mel = from_inv_mel
        self.init_noise_scale = init_noise_scale  # emprical, need to tune

        self.use_disc_loss = use_disc_loss
        self.disc_mask_order = disc_mask_order

        self.use_fft_mag_loss = use_fft_mag_loss
        self.use_log_mel_loss = use_log_mel_loss

        self.mag_power = mag_power

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
                convnext_conv_kernel_size=convnext_conv_kernel_sizes[i],
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

        if from_inv_mel:
            self.inv_mel = InverseMelScale(
                n_mels=n_mels, sample_rate=sampling_rate, n_fft=mel_n_fft
            )
            self.ifft = ISTFT(n_fft=mel_n_fft, hop_length=mel_hop_length)

        if use_disc_loss:
            # Use a smaller model as discriminator
            self.disc_estimators = nn.ModuleList([
                AudioConvNeXt(
                    n_fft=n_ffts[i],
                    hop_length=hop_lengths[i],
                    cond_channels=mel_enc_channels,
                    mel_hop_length=mel_hop_length,
                    convnext_channels=convnext_channels[i],
                    convnext_num_layers=convnext_num_layers[i] // 2,
                    num_outputs=1,
                    use_t=False,
                    use_dest_t=False,
                    analytic=False,
                )
                for i in range(self.num_branches)
            ])

        if use_fft_mag_loss:
            self.fft = STFT(n_fft=mel_n_fft, hop_length=mel_hop_length)

        if mag_power > 1:
            self.post_fft = STFT(n_fft=512, hop_length=256)
            self.post_ifft = ISTFT(n_fft=512, hop_length=256)

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
        forward_disc: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if forward_disc:
            assert self.use_disc_loss
        estimators = self.estimators if not forward_disc else self.disc_estimators
        branch_outputs = torch.stack([
            est(audio=x, audio_lens=audio_lens, mel=cond) for est in estimators
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

        output = output[:, 0]

        if self.mag_power > 1:
            eps = 1e-6
            fft, _ = self.post_fft(output, audio_lens)
            mag = (fft.real ** 2 + fft.imag ** 2 + eps).sqrt()
            fft = fft * (1.0 - (-mag).exp())
            output = self.post_ifft(fft)

        return output

    def get_disc_loss(
        self,
        x1: torch.Tensor,
        x1_pred: torch.Tensor,
        audio_lens: torch.Tensor,
        cond: torch.Tensor,
        branch_drop_rate: float = 0.0,
    ) -> torch.Tensor:
        """Get discriminator-related loss, i.e. mask-prediction loss."""

        # generate mask used to combine x1 and x1_pred
        batch_size, time = x1.shape
        kwargs = {'device': x1.device}
        # batch_size, (sin vs. cos)
        r = torch.randn(2, batch_size, self.disc_mask_order, 1, **kwargs)
        time_range = torch.arange(time, **kwargs) * (math.pi / time)
        order_range = torch.arange(1, self.disc_mask_order + 1, **kwargs).unsqueeze(-1)

        mask = (r[0] * (time_range * order_range).sin()
                + r[1] * (time_range * order_range).cos())  # (batch_size, order, time)
        mask = mask.sum(dim=1)  # (batch_size, time)
        # get a lava-lamp-like mask that's quite balanced between zeros and ones
        # mask = (mask > 0).to(torch.float)
        mask = (mask * 5.0).sigmoid()  # a soft mask in range [0, 1]

        # combined_x1 is a combined audio where some parts come from real x1
        # and some come from x1_pred which is the model output
        combined_x1 = (mask * x1) + (1.0 - mask) * x1_pred
        # reverse grad before giving it to the discriminator, as we want the
        # model to "beat" the discriminator.
        combined_x1 = ScaleGrad.apply(combined_x1, -1)

        disc_output = self.process_model(
            x=combined_x1,
            audio_lens=audio_lens,
            cond=cond,
            branch_drop_rate=branch_drop_rate,
            forward_disc=True,
        )  # (batch_size, time)
        # disc_loss is also a MSE loss
        disc_loss = self.compute_loss(
            pred=disc_output,
            target=mask,
            audio_lens=audio_lens,
            mel_scaling_loss=False,
        )
        return disc_loss

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        inv_noise: torch.Tensor,
        mel_scaling_loss: bool = True,
        branch_drop_rate: float = 0.0,
        mix_noise_scale: float = 0.2,
        eps: float = 1e-8,
    ) -> Tuple[Tensor]:
        x1 = audio
        x0 = inv_noise

        # input condition
        mel_spec = self.mel(audio)  # (batch, n_mels, time)
        cond = self.mel_encoder(mel_spec.sqrt())

        if not self.use_disc_loss:
            x1_pred = self.process_model(
                x=x0,
                audio_lens=audio_lens,
                cond=cond,
                branch_drop_rate=branch_drop_rate,
            )
            # compute losses
            main_loss = self.compute_loss(
                pred=x1_pred,
                target=x1,
                audio_lens=audio_lens,
                mel_scaling_loss=mel_scaling_loss,
                mel_spec=mel_spec,
            )

            if self.use_log_mel_loss:
                log_mel_loss = self.compute_log_mel_loss(
                    pred=x1_pred, target=x1, audio_lens=audio_lens
                )
            else:
                log_mel_loss = 0.0

            if self.use_fft_mag_loss:
                fft_mag_loss = self.compute_fft_mag_loss(
                    pred=x1_pred, target=x1, audio_lens=audio_lens
                )
            else:
                fft_mag_loss = 0.0

            disc_loss = 0.0
        else:
            batch_size = audio.shape[0]
            # add real random noise to inv_noise
            if self.from_inv_mel:
                random_noise = self.reconstuct_audio_with_random_phase(mel_spec)
                random_noise = convert_length(random_noise, audio.shape[-1])
            else:
                # scale x0 by x1's std in training
                random_noise = torch.randn_like(audio) * self.init_noise_scale
            # use sqrt to keep the noise-energy correct
            x0_mixed = random_noise * math.sqrt(mix_noise_scale) + x0 * math.sqrt(1.0 - mix_noise_scale)

            x1_pred = self.process_model(
                x=torch.cat([x0, x0_mixed], dim=0),
                audio_lens=audio_lens.repeat(2),
                cond=cond.repeat(2, 1, 1),
                branch_drop_rate=branch_drop_rate,
            )  # (batch_size * 2, time)
            # compute losses
            main_loss = self.compute_loss(
                pred=x1_pred[:batch_size],  # first half
                target=x1,
                audio_lens=audio_lens,
                mel_scaling_loss=mel_scaling_loss,
                mel_spec=mel_spec,
            )
            disc_loss = self.get_disc_loss(
                x1=x1,
                x1_pred=x1_pred[batch_size:],  # second half
                audio_lens=audio_lens,
                cond=cond.detach(),  # TODO: not sure whether we need detach here
                branch_drop_rate=branch_drop_rate,
            )
            # TODO:
            log_mel_loss, fft_mag_loss = 0.0, 0.0

        loss = (main_loss, log_mel_loss, fft_mag_loss, disc_loss)
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

    def compute_log_mel_loss(
        self, pred: Tensor, target: Tensor, audio_lens: Tensor
    ) -> Tensor:
        """Compute l1-loss on log-mel"""
        pred = safe_log(self.mel(pred))  # (batch, n_mels, time)
        target = safe_log(self.mel(target))  # (batch, n_mels, time)
        mel_spec_lens = 1 + torch.div(audio_lens, self.mel.hop_length, rounding_mode="floor")
        assert pred.shape[2] == mel_spec_lens.max().item()
        pad_mask = make_pad_mask(mel_spec_lens).logical_not().unsqueeze(1)  # (batch, 1, time)
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        loss = (loss * pad_mask).sum() / (pad_mask.sum() * pred.shape[1])
        return loss

    def compute_fft_mag_loss(
        self, pred: Tensor, target: Tensor, audio_lens: Tensor
    ) -> Tensor:
        """Compute l1-loss on fft magnitude"""
        pred, fft_lens = self.fft(pred, audio_lens)  # (batch, n_fft // 2 + 1, time)
        target, _ = self.fft(target, audio_lens)  # (batch, n_fft // 2 + 1, time)
        pad_mask = make_pad_mask(fft_lens).logical_not().unsqueeze(1)  # (batch, 1, time)
        loss = F.smooth_l1_loss(pred.abs(), target.abs(), reduction='none')
        loss = (loss * pad_mask).sum() / (pad_mask.sum() * pred.shape[1])
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
