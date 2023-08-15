#!/usr/bin/env python3
# Copyright    2023       Xiaomi Corp.        (authors: Daniel Povey)
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


from typing import List

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence


class Image2TextModel(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 encoer_dim: int,
                 decoder_dim: int,
                 vocab_size: int, ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, vocab_size)

    def forward(self,
                images: Tensor,
                labels: List[List[int]],
                sos_eos_id: int = 0):
        """
        Compute array of log-probs

        Args:
            images: (batch, in_channel, H, W)
            labels: A list-of-list IDs. Each sublist contains IDs for an utterance.
              The IDs can be either phone IDs or word piece IDs.
            sos_eos_id: sos id and eos id
        Returns:
           a Tensor containing the log-probs for each label, of shape (batch_size, seq_len+1).
        """
        assert images.shape[0] == len(labels), (images.shape[0], len(labels))

        # Forward image encoder
        image_features = self.encoder(images)  # (H * W, batch, encoder_dim)

        # Prepare the input and output labels
        ys_in = [[sos_eos_id] + utt for utt in labels]
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in = pad_sequence(ys_in, batch_first=True, padding_value=sos_eos_id)

        ys_out = [utt + [sos_eos_id] for utt in labels]
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out = pad_sequence(ys_out, batch_first=True, padding_value=sos_eos_id)

        device = images.device
        ys_in = ys_in.to(device)  # (batch, seq_len+1)
        ys_out = ys_out.to(device)  # (batch, seq_len+1)

        src_key_padding_mask = (ys_in == sos_eos_id)  # (batch, seq_len+1)
        # The first token is sos_id
        src_key_padding_mask[:, 0] = False

        label_embed = self.token_embed(ys_in.transpose(0, 1))  # (seq_len+1, batch, decoder_dim)
        decoder_embedding = self.decoder(
            label_embed, memory=image_features, src_key_padding_mask=src_key_padding_mask)
        # decoder_embedding: (seq_len+1, batch, decoder_dim)

        logits = self.out_proj(decoder_embedding).transpose(0, 1)  # (batch, seq_len+1, vocab_size)

        log_probs = logits.log_softmax(dim=-1)
        log_probs = torch.gather(log_probs, dim=-1, index=ys_out.unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs.masked_fill(src_key_padding_mask, 0.0)
        # log_probs: (batch_size, seq_len+1)

        return log_probs


