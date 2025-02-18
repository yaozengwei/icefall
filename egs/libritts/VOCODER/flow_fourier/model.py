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
from typing import Optional, Tuple, Union

import torch
from torchaudio.transforms import MelSpectrogram
from torch import Tensor, nn
from icefall.utils import make_pad_mask
from audio_utils import (
    ISTFT,
    InverseMelScale,
    analytic_transform,
    convert_length,
    remove_negative_frequency,
    safe_log,
)
from convnext import AudioConvNeXt, MelEncoder


def sample_t_and_dest_t(
    x0: Tensor,
    x1: Tensor,
    audio_lens: Tensor,
    mel_spec: Tensor,
    use_aux_loss: bool,
    skipping_levels: int = 8,
    t_sigma: float = 0.1,
    consistency_fraction: float = 0.25,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Union[Tensor, float], int]:
    """
    Sample time values t and dest_t; possibly reorder and duplicate some
    batch elements of x with the same t and different dest_t values.

    Args:
       x: the batch of real images, of shape (batch_size, num_channels, height, width)
      use_aux_loss: true if on this batch we will be computing auxiliary loss values
          involving dest_t.

    Returns: (t, dest_t, x0_mod, x1_mod, loss_scale, num_dup), where:
      t: (modified_batch_size, 1, 1, 1), a Tensor of "time" (t) values with 0 <= t <= 1.
      dest_t: (modified_batch_size, 1, 1, 1), a Tensor of "destination-time" (dest_t)
         values with 0 <= t <= 1.
      x0_mod, x1_mod: (modified_batch_size, num_channels, height, width), if use_aux_loss
         is True the batch will be reordered and some elements duplicated.
      loss_scale: If use_aux_loss is False this will be 1.0; otherwise a Tensor
         of shape (modified_batch_size, 1, 1, 1), containing 0.5 for (reordered) batch
         elements which are duplicated.  This is to prevent these images getting
         larger-than-normal scale in the main loss.
      num_dup: Contains the number of elements in the batch that are duplicated,
         for purposes of computing the consistency loss; the returned batch will
         be arranged as:
              num_dup + num_dup + (orig_batch_size - num_dup)
         where the first "num_dup" elements have the originally sampled delta_t,
         the next "num_dup" elements are repeats of the first elements but with
         half the originally sampled delta_t.
    """
    def get_time_shape(x):
        # returns a narrowed version of x with the same shape as the need the time to
        # be, i.e. (batch_size, 1, 1, 1)
        for n in range(1, x.ndim):
            x = x.narrow(n, 0, 1)
        return x

    t = torch.rand_like(get_time_shape(x0.real if x0.is_complex() else x0))

    batch_size = t.shape[0]

    t_power = torch.arange(batch_size, device=t.device) % skipping_levels  # 0,1,2,..,6,7,0,1,2..
    delta_t = 0.5 ** t_power.reshape(*t.shape)   # delta_t: (1.0, 0.5, 0.25, 1.0, 0.5, 0.25)
    if use_aux_loss:
        # Round t down to various powers of 1/2.  Eventually these powers are so large
        # that this has little effect, and we'll add random noise later so we'll still get
        # values arbitrarily close to 1.
        t = t - torch.remainder(t, delta_t)  # generates t values in {0}, {0, 0.5}, {0, 0.25, 0.5, 0.75}, {0, 1/128, 2/128... }
    else:
        # if we're not using the aux loss, delta_t matters much less, but we
        # still try to have a fairly similar distribution of delta_t to what we
        # have in normal inference, so the model still learns to generate its
        # standard output well regardless of the value of delta_t.
        delta_t = torch.min(delta_t, (1.0 - t))

    dest_t = t + delta_t

    def reflect_on_edges(x):
        return x - 2.0 * (x - x.clamp(min=0, max=1))

    # randomize t and dest_t by small random amounts.
    t = reflect_on_edges(t + t_sigma * torch.randn_like(t))
    dest_t = reflect_on_edges(dest_t + t_sigma * torch.randn_like(t))

    if use_aux_loss:
        # This block takes care of the logic for duplicating some elements of the
        # batch so that we can compute the "consistency loss" for a small subset
        # of elements.
        eps = 1.0e-20
        # this 'power' relates to logic for sampling with more probability when
        # delta_t is large, but we don't want to do this too strongly as we do
        # need to ensure consistency for small delta_t as well.
        power = 0.5
        log_abs_delta_t = power * torch.log(eps + (dest_t - t).abs())
        gumbel = -torch.log(eps - torch.log(eps + torch.rand_like(t)))
        delta_t_for_sampling = gumbel + log_abs_delta_t
        # The gumbel trick works for a single sample.  For multiple samples
        # I'm not sure if this is still true or what are the exact marginal probabilities
        # of sampling different elements, but they key thing here is to
        # give more probability on sampling values with large delta_t, because
        # these are the values where the "consistency loss" is going to be more
        # challenging to optimize.
        _values, indexes = torch.sort(delta_t_for_sampling.flatten(),
                                      descending=True)
        num_dup = int(consistency_fraction * batch_size + 0.999)
        assert num_dup > 0
        dup = indexes[:num_dup]
        remaining = indexes[num_dup:]
        x0_dup, x0_remaining = x0[dup], x0[remaining]
        x0 = torch.cat((x0_dup, x0_dup, x0_remaining), dim=0)
        x1_dup, x1_remaining = x1[dup], x1[remaining]
        x1 = torch.cat((x1_dup, x1_dup, x1_remaining), dim=0)
        t_dup, t_remaining = t[dup], t[remaining]
        audio_lens_dup, audio_lens_remaining = audio_lens[dup], audio_lens[remaining]
        audio_lens = torch.cat((audio_lens_dup, audio_lens_dup, audio_lens_remaining), dim=0)
        mel_spec_dup, mel_spec_remaining = mel_spec[dup], mel_spec[remaining]
        mel_spec = torch.cat((mel_spec_dup, mel_spec_dup, mel_spec_remaining), dim=0)
        half_scale = torch.full_like(t_dup, 0.5)
        full_scale = torch.full_like(t_remaining, 1.0)
        loss_scale = torch.cat((half_scale, half_scale, full_scale), dim=0)
        dest_t_dup, dest_t_remaining = dest_t[dup], dest_t[remaining]
        t = torch.cat((t_dup, t_dup, t_remaining), dim=0)
        dest_t = torch.cat((dest_t_dup, 0.5 * (t_dup + dest_t_dup), dest_t_remaining),
                           dim=0)
    else:
        loss_scale = 1.0
        num_dup = 0

    if random.random() < 0.01:
        logging.info(
            f"t = {t.flatten()}, delta_t = {(dest_t - t).flatten()}, "
            f"loss_scale = {1.0 if isinstance(loss_scale, float) else loss_scale.flatten()}, "
            f"num_dup={num_dup}"
        )
    return t, dest_t, x0, x1, audio_lens, mel_spec, loss_scale, num_dup


class Vocoder(nn.Module):
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
        higher_order: bool = False,
        analytic: bool = False,
        from_inv_mel: bool = True,
        init_noise_scale: float = 0.1,
        mag_power: int = 1,
    ):
        super().__init__()
        self.num_branches = len(n_ffts)
        assert len(hop_lengths) == self.num_branches
        assert len(convnext_num_layers) == self.num_branches
        assert len(convnext_channels) == self.num_branches
        assert len(convnext_conv_kernel_sizes) == self.num_branches
        self.higher_order = higher_order
        self.num_outputs = 1 if not higher_order else 3
        self.analytic = analytic
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
                convnext_conv_kernel_size=convnext_conv_kernel_sizes[i],
                num_outputs=self.num_outputs,
                use_dest_t=higher_order,
                analytic=analytic,
                mag_power=mag_power,
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
        t: Tensor,
        dest_t: Optional[Tensor] = None,
        branch_drop_rate: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.higher_order:
            assert dest_t is not None

        branch_outputs = torch.stack([
            estimator(
                audio=x,
                audio_lens=audio_lens,
                t=t.flatten(),
                dest_t=dest_t.flatten() if self.higher_order else None,
                mel=cond,
            )
            for estimator in self.estimators
        ], dim=1)  # (batch, num_branches, num_output, time)

        # fuse all branches
        if not self.training or branch_drop_rate <= 0:
            output = branch_outputs.mean(dim=1)
        else:
            if random.random() < 0.05:
                logging.info(f"branch_drop_rate={branch_drop_rate}")
            # At the start of training, apply random branch masking
            mask = torch.rand(branch_outputs.shape[:2], device=x.device) > branch_drop_rate
            mask[mask.sum(dim=1) == 0] = True  # Unmask if all branches are dropped
            weight = (mask / mask.sum(dim=1, keepdim=True))[:, :, None, None]  # (batch, num_branches, 1, 1)
            output = (branch_outputs * weight).sum(dim=1)

        if not self.higher_order:
            vt = output[:, 0]
            return vt, None, None
        else:
            # output: (batch, num_output, time)
            vt, order2_mid, order2_far = output[:, :3].unbind(dim=1)
            delta_t = dest_t - t
            x_dest = x + delta_t * vt + (delta_t ** 2) * (order2_mid + order2_far)
            half_delta_t = delta_t * 0.5
            x_mid = x + half_delta_t * vt + (half_delta_t ** 2) * order2_mid
            return vt, x_mid, x_dest

    def forward(
        self,
        ema_model: nn.Module,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        mel_scaling_loss: bool = True,
        branch_drop_rate: float = 0.0,
        use_aux_loss: bool = True,
        skipping_levels: int = 8,
        t_sigma: float = 0.1,
        consistency_fraction: float = 0.25,
        close_loss_decay: float = 0.2,
        eps: float = 1e-8,
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

        if self.analytic:
            x1 = analytic_transform(x1)
            x0 = analytic_transform(x0)
            # Now x1 and x0 are both complex tensors

        if self.higher_order:
            (
                t, dest_t, x0, x1, audio_lens, mel_spec, main_loss_scale, num_dup
            ) = sample_t_and_dest_t(
                x0=x0,
                x1=x1,
                audio_lens=audio_lens,
                mel_spec=mel_spec,
                use_aux_loss=use_aux_loss,
                skipping_levels=skipping_levels,
                t_sigma=t_sigma,
                consistency_fraction=consistency_fraction,
            )
            delta_t = dest_t - t
        else:
            t = torch.rand([audio.shape[0], 1], device=audio.device, dtype=audio.dtype)

        xt = (1.0 - t) * x0 + t * x1
        ut = x1 - x0

        cond = self.mel_encoder(mel_spec.sqrt())
        # if batch_idx_train <= 2000:
        #     drop_rate = 0.1 * (1 - batch_idx_train / 2000)
        # else:
        #     drop_rate = 0.0
        vt, x_mid, x_dest = self.process_model(
            x=xt,
            audio_lens=audio_lens,
            cond=cond,
            t=t,
            dest_t=dest_t if self.higher_order else None,
            branch_drop_rate=branch_drop_rate,
        )
        # shape of vt, x_mid, x_dest should be: (batch_size, time)

        # compute losses
        main_loss = self.compute_loss(
            pred=vt,
            target=ut,
            audio_lens=audio_lens,
            loss_scale=main_loss_scale if self.higher_order else 1.0,
            mel_scaling_loss=mel_scaling_loss,
            mel_spec=mel_spec,
        )

        if use_aux_loss and self.higher_order:
            # first compute ref_loss which encourages the model outputs to stay similar to the
            # moving-average model's.  This helps reduce parameter noise.
            with torch.no_grad():
                ref_vt, ref_x_mid, ref_x_dest = ema_model.process_model(
                    x=xt,
                    audio_lens=audio_lens,
                    cond=cond,
                    t=t,
                    dest_t=dest_t,
                    branch_drop_rate=0.0,
                )
            ref_loss1 = self.compute_loss(
                pred=x_dest,
                target=ref_x_dest,
                audio_lens=audio_lens,
                loss_scale=1.0,
                mel_scaling_loss=mel_scaling_loss,
                mel_spec=mel_spec,
            )
            ref_loss2 = self.compute_loss(
                pred=vt,
                target=ref_vt,
                audio_lens=audio_lens,
                loss_scale=1.0,
                mel_scaling_loss=mel_scaling_loss,
                mel_spec=mel_spec,
            )

            with torch.no_grad():
                # get a better estimate of dest_xt by breaking the interval into two halves at mid_t.
                mid_t = 0.5 * (t + dest_t)
                vt_MID, _x_mid_MID, x_dest_MID = self.process_model(
                    x=x_mid,
                    audio_lens=audio_lens,
                    cond=cond,
                    t=mid_t,
                    dest_t=dest_t,
                    branch_drop_rate=0.0,
                )
            # main auxiliary loss, use mid_x_dest as more-accurate target for
            # x_dest.  the denominator can be thought of as a heuristic scale to
            # balance the magnitudes of smaller and larger delta_t
            # regions.  the denominator can be thought of as a heuristic scale to balance the
            # magnitudes of smaller and larger delta_t, and also scale down loss for t values close
            # to 1 which would otherwise dominate the loss.
            dest_loss_scale = 1.0 / (delta_t ** 2 + eps)
            dest_loss = self.compute_loss(
                pred=x_dest,
                target=x_dest_MID,
                audio_lens=audio_lens,
                loss_scale=dest_loss_scale,
                mel_scaling_loss=mel_scaling_loss,
                mel_spec=mel_spec,
            )

            # a subset of batch elements are duplicated and have the same t value but different
            # dest_t values.  the first 'num_dup' elements have half the delta_t values of the
            # next 'num_dup' elements, so the dest_xt of the next num_dup elements
            # should be equal to the x_mid of the first 'num_dup' elements.
            x_mid_dup = x_mid[:num_dup]
            x_mid_dup_accurate = x_dest_MID[num_dup:2 * num_dup]  # note: has no grad.
            delta_t_dup = delta_t[num_dup:2 * num_dup]

            consistency_loss_scale = 1.0 / (delta_t_dup ** 2 + eps)
            consistency_loss = self.compute_loss(
                pred=x_mid_dup,
                target=x_mid_dup_accurate,
                audio_lens=audio_lens[:num_dup],
                loss_scale=consistency_loss_scale,
                mel_scaling_loss=mel_scaling_loss,
                mel_spec=mel_spec[:num_dup]
            )

            half_delta_t = delta_t * 0.5
            x_mid_fromvt = xt + (0.5 * (vt.detach() + vt_MID)) * half_delta_t

            close_loss_scale = (half_delta_t.abs() / -close_loss_decay).exp() / (half_delta_t ** 2 + eps)  # larger eps.
            close_loss = self.compute_loss(
                pred=x_mid,
                target=x_mid_fromvt,
                audio_lens=audio_lens,
                loss_scale=close_loss_scale,
                mel_scaling_loss=mel_scaling_loss,
                mel_spec=mel_spec,
            )

            loss = (main_loss, dest_loss, consistency_loss, ref_loss1, ref_loss2, close_loss)
        else:
            loss = (main_loss,)

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

        if self.analytic:
            batch, time = pred.shape
            pred = torch.view_as_real(pred).permute(2, 0, 1).reshape(2 * batch, time)
            target = torch.view_as_real(target).permute(2, 0, 1).reshape(2 * batch, time)
            audio_lens = audio_lens.repeat(2)
            if isinstance(loss_scale, Tensor):
                loss_scale = loss_scale.repeat(2, 1)
            if mel_spec is not None:
                mel_spec = mel_spec.repeat(2, 1, 1)

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

        if self.analytic:
            noise = analytic_transform(noise)
            # Now noise is complex

        cond = self.mel_encoder(mel_spec.sqrt())

        # use fixed euler solver for ODEs.
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=noise.device)
        t, dt = t_span[0], t_span[1] - t_span[0]
        x = noise
        batch_size = x.shape[0]
        for step in range(1, len(t_span)):
            if not self.higher_order:
                vt, _, _ = self.process_model(
                    x=x,
                    audio_lens=audio_lens,
                    cond=cond,
                    t=t[None, None].expand(batch_size, 1),
                )
                x = x + vt * dt
            else:
                dest_t = t + dt
                _, _, x_dest = self.process_model(
                    x=x,
                    audio_lens=audio_lens,
                    cond=cond,
                    t=t[None, None].expand(batch_size, 1),
                    dest_t=dest_t[None, None].expand(batch_size, 1),
                )
                x = x_dest

            if self.analytic:
                # remove negative frequency at each step
                x = remove_negative_frequency(x)

            t = t_span[step]

        pred_audio = x if not self.analytic else x.real
        if clamp_pred:
            pred_audio = pred_audio.clamp(min=-1.0, max=1.0)

        if log_mel_diff:
            mel_spec_noise = self.mel(noise if not self.analytic else noise.real)  # (batch, n_mels, time)
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

    def inverse_infer(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        n_timesteps: int = 8,
        start_t: float = 0.999,
    ) -> Tensor:
        assert not self.analytic and not self.higher_order

        mel_spec = self.mel(audio)  # (batch, n_mels, time)

        # sample noise p(x_0)
        if self.from_inv_mel:
            noise = self.reconstuct_audio_with_random_phase(mel_spec)
            noise = convert_length(noise, audio.shape[-1])
        else:
            # scale x0 by x1's std in training
            noise = torch.randn_like(audio) * self.init_noise_scale

        # start point
        x = audio * start_t + noise * (1.0 - start_t)

        cond = self.mel_encoder(mel_spec.sqrt())

        # use fixed euler solver for ODEs.
        t_span = torch.linspace(start_t, 0, n_timesteps + 1, device=noise.device)
        t, dt = t_span[0], t_span[1] - t_span[0]
        batch_size = x.shape[0]
        for step in range(1, len(t_span)):
            vt, _, _ = self.process_model(
                x=x,
                audio_lens=audio_lens,
                cond=cond,
                t=t[None, None].expand(batch_size, 1),
            )
            x = x + vt * dt
            t = t_span[step]

        inv_noise = x

        return inv_noise, noise
