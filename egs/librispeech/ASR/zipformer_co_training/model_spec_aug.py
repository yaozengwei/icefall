# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
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

import math
import random
from typing import List, Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from lhotse.dataset import SpecAugment
from lhotse.dataset.signal_transforms import mask_along_axis_optimized
from lhotse.dataset.signal_transforms import time_warp as time_warp_impl

from icefall.utils import add_sos, make_pad_mask


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.vocab_size = vocab_size

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward_old(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss

    def forward2(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert x.size(0) == x_lens.size(0) == len(y), (x.shape, x_lens.shape, len(y))
        batch_size = x.size(0)

        # Compute encoder outputs on the duplicated batch
        encoder_out, encoder_out_lens = self.forward_encoder(
            x.repeat(2, 1, 1), x_lens.repeat(2)
        )  # (N * 2, T, C), (N * 2)

        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N * 2, T, C)

        # Perform ctc-decoding to get output labels
        hyps = ctc_greedy_search(ctc_output, encoder_out_lens)  # N * 2 utterances
        targets = []
        target_lengths = []
        for utt in (y + y + hyps[batch_size:] + hyps[:batch_size]):
            targets += utt
            target_lengths.append(len(utt))

        ctc_output = ctc_output.repeat(2, 1, 1)  # (N * 4, T, C)
        encoder_out_lens = encoder_out_lens.repeat(2)  # (N * 4,)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N * 4, C)
            targets=torch.tensor(targets),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=torch.tensor(target_lengths),
            reduction="none",
        )  # (N * 4,)

        assert ctc_loss.shape == (batch_size * 4,)
        ctc_loss_label = ctc_loss[:batch_size * 2].sum() / 2.0
        ctc_loss_cosub = ctc_loss[batch_size * 2:].sum() / 2.0

        return ctc_loss_label, ctc_loss_cosub

    def forward_spec_aug_old(
        # def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: List[List[str]],
        use_spec_aug: bool = False,
        supervision_segments: Optional[torch.Tensor] = None,
        spec_augment: Optional[SpecAugment] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert x.size(0) == x_lens.size(0) == len(y), (x.shape, x_lens.shape, len(y))
        batch_size = x.size(0)

        if use_spec_aug:
            assert spec_augment is not None
            assert supervision_segments is not None
            x1 = spec_augment(x, supervision_segments=supervision_segments)
            x2 = spec_augment(x, supervision_segments=supervision_segments)
            x = torch.cat([x1, x2], dim=0)
        else:
            x = x.repeat(2, 1, 1)

        x_lens = x_lens.repeat(2)

        # Compute encoder outputs on the duplicated batch
        encoder_out, encoder_out_lens = self.forward_encoder(
            x, x_lens
        )  # (N * 2, T, C), (N * 2)

        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N * 2, T, C)

        # Get frame-level labels
        cosub_labels = ctc_output.argmax(dim=2)  # (N * 2, T)
        # Exchange the labels
        cosub_labels = torch.cat(
            [cosub_labels[batch_size:], cosub_labels[:batch_size]], dim=0
        )
        # Compute cosub loss
        ignore_id = -100
        length_mask = make_pad_mask(encoder_out_lens)
        cosub_labels.masked_fill_(length_mask, ignore_id)
        cosub_ce_loss = nn.functional.nll_loss(
            input=ctc_output.view(-1, self.vocab_size),
            target=cosub_labels.view(-1),
            ignore_index=ignore_id,
            reduction="sum",
        )

        # Compute CTC loss
        targets = []
        target_lengths = []
        for utt in (y + y):
            targets += utt
            target_lengths.append(len(utt))

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N * 2, C)
            targets=torch.tensor(targets),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=torch.tensor(target_lengths),
            reduction="sum",
        )

        return ctc_loss / 2.0, cosub_ce_loss / 2.0

    def forward_kl(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert x.size(0) == x_lens.size(0) == len(y), (x.shape, x_lens.shape, len(y))
        batch_size = x.size(0)

        # Compute encoder outputs on the duplicated batch
        encoder_out, encoder_out_lens = self.forward_encoder(
            x.repeat(2, 1, 1), x_lens.repeat(2)
        )  # (N * 2, T, C), (N * 2)

        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N * 2, T, C)

        # Exchange the targets
        cosub_targets = torch.cat(
            [ctc_output[batch_size:], ctc_output[:batch_size]], dim=0
        )
        # Compute cosub loss
        length_mask = make_pad_mask(encoder_out_lens)
        cosub_kl_loss = nn.functional.kl_div(
            input=ctc_output,
            target=cosub_targets,
            reduction="none",
            log_target=True,
        )  # (N * 2, T, C)
        cosub_kl_loss = cosub_kl_loss.masked_fill_(length_mask.unsqueeze(-1), 0.0)
        cosub_kl_loss = cosub_kl_loss.sum()

        # Compute CTC loss
        targets = []
        target_lengths = []
        for utt in (y + y):
            targets += utt
            target_lengths.append(len(utt))

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N * 2, C)
            targets=torch.tensor(targets),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=torch.tensor(target_lengths),
            reduction="sum",
        )

        return ctc_loss / 2.0, cosub_kl_loss / 2.0

    # def forward_spec_aug_new(
    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: List[List[str]],
        use_spec_aug: bool = False,
        use_time_warp: bool = False,
        use_time_mask: bool = False,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
        num_frame_masks: int = 10,
        features_mask_size: int = 27,
        num_feature_masks: int = 2,
        frames_mask_size: int = 100,
        max_frames_mask_fraction: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert x.size(0) == x_lens.size(0) == len(y), (x.shape, x_lens.shape, len(y))
        batch_size = x.size(0)

        if use_spec_aug:
            if use_time_warp:
                assert supervision_segments is not None
                # Apply time warping before duplicating
                x = time_warp(
                    x,
                    time_warp_factor=time_warp_factor,
                    supervision_segments=supervision_segments,
                )

            # Apply frequency masking on two copies respectively
            x1 = frequency_mask(
                x,
                features_mask_size=features_mask_size,
                num_feature_masks=num_feature_masks,
            )
            x2 = frequency_mask(
                x,
                features_mask_size=features_mask_size,
                num_feature_masks=num_feature_masks,
            )

            if use_time_mask:
                # Apply time masking on two copies respectively
                x1 = time_mask(
                    x1,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size,
                    max_frames_mask_fraction=max_frames_mask_fraction,
                )
                x2 = time_mask(
                    x2,
                    num_frame_masks=num_frame_masks,
                    frames_mask_size=frames_mask_size,
                    max_frames_mask_fraction=max_frames_mask_fraction,
                )

            x = torch.cat([x1, x2], dim=0)
        else:
            x = x.repeat(2, 1, 1)

        x_lens = x_lens.repeat(2)

        # Compute encoder outputs on the duplicated batch
        encoder_out, encoder_out_lens = self.forward_encoder(
            x, x_lens
        )  # (N * 2, T, C), (N * 2)

        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N * 2, T, C)

        # Get frame-level labels
        cosub_labels = ctc_output.argmax(dim=2)  # (N * 2, T)
        # Exchange the labels
        cosub_labels = torch.cat(
            [cosub_labels[batch_size:], cosub_labels[:batch_size]], dim=0
        )
        # Compute cosub loss
        ignore_id = -100
        length_mask = make_pad_mask(encoder_out_lens)
        cosub_labels.masked_fill_(length_mask, ignore_id)
        cosub_ce_loss = nn.functional.nll_loss(
            input=ctc_output.view(-1, self.vocab_size),
            target=cosub_labels.view(-1),
            ignore_index=ignore_id,
            reduction="sum",
        )

        # Compute CTC loss
        targets = []
        target_lengths = []
        for utt in (y + y):
            targets += utt
            target_lengths.append(len(utt))

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N * 2, C)
            targets=torch.tensor(targets),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=torch.tensor(target_lengths),
            reduction="sum",
        )

        return ctc_loss / 2.0, cosub_ce_loss / 2.0


def frequency_mask(
    features: torch.Tensor,
    p: float = 0.9,
    features_mask_size: int = 27,
    num_feature_masks: int = 10,
):
    assert len(features.shape) == 3, (
        "SpecAugment only supports batches of single-channel feature matrices."
    )
    features = features.clone()
    for sequence_idx in range(features.size(0)):
        if random.random() > p:
            # Randomly choose whether this transform is applied
            continue
        feat = features[sequence_idx]
        mean = feat.mean()
        # Frequency masking
        feat = mask_along_axis_optimized(
            feat,
            mask_size=features_mask_size,
            mask_times=num_feature_masks,
            mask_value=mean,
            axis=2,
        )
        features[sequence_idx] = feat

    return features


def time_mask(
    features: torch.Tensor,
    p: float = 0.9,
    num_frame_masks: int = 10,
    frames_mask_size: int = 100,
    max_frames_mask_fraction: float = 0.15,
):
    assert len(features.shape) == 3, (
        "SpecAugment only supports batches of single-channel feature matrices."
    )
    features = features.clone()
    for sequence_idx in range(features.size(0)):
        if random.random() > p:
            # Randomly choose whether this transform is applied
            continue
        feat = features[sequence_idx]
        mean = feat.mean()
        # Time masking
        max_tot_mask_frames = max_frames_mask_fraction * feat.size(0)
        num_frame_masks = min(
            num_frame_masks,
            math.ceil(max_tot_mask_frames / frames_mask_size),
        )
        max_mask_frames = min(
            frames_mask_size, max_tot_mask_frames // num_frame_masks
        )
        feat = mask_along_axis_optimized(
            feat,
            mask_size=max_mask_frames,
            mask_times=num_frame_masks,
            mask_value=mean,
            axis=1,
        )
        features[sequence_idx] = feat

    return features


def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    supervision_segments: Optional[torch.Tensor] = None,
):
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert len(features.shape) == 3, (
        "SpecAugment only supports batches of single-channel feature matrices."
    )
    features = features.clone()
    if supervision_segments is None:
        # No supervisions - apply spec augment to full feature matrices.
        for sequence_idx in range(features.size(0)):
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            features[sequence_idx] = time_warp_impl(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        # Supervisions provided - we will apply time warping only on the supervised areas.
        for sequence_idx, start_frame, num_frames in supervision_segments:
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            end_frame = start_frame + num_frames
            features[sequence_idx, start_frame:end_frame] = time_warp_impl(
                features[sequence_idx, start_frame:end_frame], factor=time_warp_factor
            )

    return features


def ctc_greedy_search(
    ctc_probs: torch.Tensor, encoder_out_lens: torch.Tensor
) -> List[List[int]]:
    """Apply CTC greedy search

    Args:
         ctc_probs (torch.Tensor): (batch, max_len, feat_dim)
         encoder_out_lens (torch.Tensor): (batch, )
    Returns:
         List[List[int]]: best path result
    """
    batch_size = ctc_probs.shape[0]
    encoder_mask = make_pad_mask(encoder_out_lens)
    maxlen = ctc_probs.size(1)

    _, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(encoder_mask, 0)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
    return hyps


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    # from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/common.py
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp
