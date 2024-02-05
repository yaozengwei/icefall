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

from typing import Tuple

import k2
import torch
import torch.nn as nn


class TtsModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        vocab_size: int,
        blank_id: int,
    ):
        """Transducer model for TTS.

        - Sequence Transduction with Recurrent Neural Networks
          (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training
          (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder:
            The transcription network, with text tokens as input.
          decoder:
            The prediction network, with audio tokens as input.
            It should contain two attribute: `blank_id` and `num_codebooks`.
          joiner:
            It has two inputs with shapes: (N, T_text, encoder_dim) and
            (N, T_audio, decoder_dim).
            Its output shape is (N, T_text, T_audio, vocab_size).
            Note that its output contains unnormalized probs,
            i.e., not processed by log-softmax.
        """
        super().__init__()
        self.blank_id = blank_id  # i.e., maximum codebook-id + 1

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = nn.Linear(encoder_dim, vocab_size)
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        prompt_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          x:
            Input text tokens, of shape (batch, text_len).
          x_lens:
            A 1-D tensor of shape (batch,). It contains the number of tokens in `x`
            before padding.
          prompt:
            Prompt tensor of shape (batch, prompt_len, prompt_dim).
          prompt_lens:
            A tensor of shape (batch,). It contains the number of frames in
            `prompt` before padding.
          y:
            Audio tokens of shape (batch, audio_len).
          y_lens:
            A 1-D tensor of shape (batch,). It contains the number of tokens in `y`
            before padding.
          prune_range:
            The prune range for rnnt loss, it means how many audio tokens (context)
            we are considering for each text token to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part.
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part.

        Returns:
          Return the transducer losses, in form of (simple_loss, pruned_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        # forward encoder
        encoder_out = self.encoder(x, x_lens, memory=prompt, memory_lens=prompt_lens)

        # Now for the decoder, i.e., the prediction module
        sos_id = self.decoder.sos_id  # i.e., maximum codebook-id + 1
        # (batch, 1 + audio_len), start with SOS
        sos_y = nn.functional.pad(y, (1, 0), value=sos_id)

        # (batch, audio_len + 1, decoder_dim)
        decoder_out = self.decoder(sos_y, y_lens + 1)

        y = y.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4), dtype=torch.int64, device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        # Note: Don't be confused by `lm` and `am` here,
        # we just keep the names as we used in ASR recipes
        lm = self.simple_lm_proj(decoder_out)  # (batch, audio_len + 1, vocab_size)
        am = self.simple_am_proj(encoder_out)  # (batch, text_len, vocab_size)

        blank_id = self.blank_id  # i.e., maximum codebook-id + 1

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [batch, text_len, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [batch, text_len, prune_range, encoder_dim]
        # lm_pruned : [batch, text_len, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [batch, text_len, prune_range, vocab_size]
        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss
