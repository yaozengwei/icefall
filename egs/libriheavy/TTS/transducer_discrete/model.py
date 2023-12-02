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

from icefall.utils import make_pad_mask
from scaling import ScaledLinear


class TtsModel(nn.Module):
    def __init__(
        self,
        text_embed: nn.Module,
        text_encoder: nn.Module,
        prompt_embed: nn.Module,
        prompt_encoder: nn.Module,
        decoder_embed: nn.Module,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 1025,
        num_codebooks: int = 4,
        blank_id: int = 0,
    ):
        """Transducer model for TTS.

        - Sequence Transduction with Recurrent Neural Networks
          (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training
          (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder:
            The transcription network, with text tokens as input.
          prompt_encoder:
            The prompt network, with prompt audio tokens as input.
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

        self.blank_id = blank_id
        self.num_codebooks = num_codebooks

        self.text_embed = text_embed
        self.text_encoder = text_encoder

        self.prompt_embed = prompt_embed
        self.prompt_encoder = prompt_encoder

        self.decoder_embed = decoder_embed
        self.decoder = decoder

        self.joiner = joiner
        self.simple_am_proj = ScaledLinear(encoder_dim, vocab_size, initial_scale=0.25)
        self.simple_lm_proj = ScaledLinear(decoder_dim, vocab_size, initial_scale=0.25)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        ys: torch.Tensor,
        y_lens: torch.Tensor,
        prompt: torch.Tensor,
        prompt_lens: torch.Tensor,
        codebook_index: int,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            Input text tokens, of shape (N, T_text).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
          ys:
            Discrete audio tokens of shape (N, T_audio, num_codebooks).
          y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `ys`
            before padding.
          prompt:
            Audio tokens as speaker prompt, of shape (N, T_prompt, num_codebooks)
          prompt_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `prompt`
            before padding.
          codebook_index:
            Codebook index we want to predict, in range of [0, num_codebooks - 1].
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
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert ys.ndim == 3, ys.shape
        assert y_lens.ndim == 1, y_lens.shape
        assert prompt.ndim == 3, prompt.shape
        assert prompt_lens.ndim == 1, prompt_lens.shape

        assert (
            x.size(0)
            == x_lens.size(0)
            == ys.size(0)
            == y_lens.size(0)
            == prompt.size(0)
            == prompt_lens.size(0)
        ), (
            x.shape,
            x_lens.shape,
            ys.shape,
            y_lens.shape,
            prompt.shape,
            prompt_lens.shape,
        )

        # forward prompt-encoder
        prompt = self.prompt_embed(prompt)
        prompt_out, prompt_out_lens = self.prompt_encoder(
            x=prompt,
            x_lens=prompt_lens,
            src_key_padding_mask=make_pad_mask(prompt_lens),
        )
        prompt_out = prompt_out.transpose(0, 1)  # (N, T, D) -> (T, N, D)

        # forward text-encoder with audio prompt output
        x_out = self.text_embed(x)
        x_out = x_out.transpose(0, 1),  # (N, T, D) -> (T, N, D)
        encoder_out, encoder_out_lens = self.text_encoder(
            x=x_out,
            x_lens=x_lens,
            src_key_padding_mask=make_pad_mask(x_lens),
            memory=prompt_out,
            memory_key_padding_mask=make_pad_mask(prompt_out_lens),
        )
        encoder_out = encoder_out.transpose(0, 1)  # (N, T, D) -> (T, N, D)

        # Now for the decoder, i.e., the prediction module
        num_codebooks = self.num_codebooks
        assert 0 <= codebook_index < num_codebooks, (codebook_index, num_codebooks)
        assert ys.shape[-1] == num_codebooks, (ys.shape, num_codebooks)

        cur_y = ys[:, :, codebook_index]  # (N, T_audio)
        blank_id = self.blank_id
        # (N, 1 + T_audio), start with SOS
        sos_cur_y = nn.functional.pad(cur_y, (1, 0), value=blank_id)

        if codebook_index > 0:
            pre_ys = ys[:, :, :codebook_index]  # (N, T_audio, codebook_index)
            # (N, T_audio + 1, codebook_index), start with SOS
            pre_ys_eos = nn.functional.pad(pre_ys, (0, 0, 0, 1), value=blank_id)
        else:
            pre_ys_eos = None

        # (N, 1 + T_audio, decoder_dim)
        decoder_in = self.decoder_embed(sos_cur_y, codebook_index, pre_ys_eos)
        # (N, 1 + T_audio, decoder_dim)
        decoder_out = self.decoder(decoder_in)

        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        # Note: Don't be confused by `lm` and `am` here,
        # we just keep the names as we used in ASR
        lm = self.simple_lm_proj(decoder_out)  # (N, 1 + T_audio, vocab_size)
        am = self.simple_am_proj(encoder_out)  # (N, T_text, vocab_size)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=cur_y,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T_text, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T_text, prune_range, encoder_dim]
        # lm_pruned : [B, T_text, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T_text, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=cur_y,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss
