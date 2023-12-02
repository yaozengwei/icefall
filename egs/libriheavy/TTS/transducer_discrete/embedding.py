# Copyright    2023  Xiaomi Corp.        (authors: Zengwei Yao)
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

import copy
from typing import Optional

import torch
import torch.nn as nn

from scaling import Balancer


class TextEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1025,
        blank_id: int = 0,
        embed_dim: int = 512,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          blank_id:
            The ID of the blank symbol.
          embed_dim:
            Dimension of the input embedding.
        """
        super().__init__()

        self.blank_id = blank_id
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=blank_id,
        )
        self.balancer = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: Text tokens, of shape (N, T).

        Returns:
          Embedding tensor, of shape (N, T, embed_dim).
        """
        # (N, T, D)
        embedding_out = self.embeddings(x)
        embedding_out = self.balancer(embedding_out)

        return embedding_out


class PromptEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1025,
        blank_id: int = 0,
        num_codebooks: int = 4,
        embed_dim: int = 512,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          blank_id:
            The ID of the blank symbol.
          num_codebooks:
            Number of codebooks used to encode audio.
          embed_dim:
            Dimension of the input embedding.
        """
        super().__init__()

        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks

        embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=blank_id,
        )
        balancer = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )
        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    copy.deepcopy(embedding),
                    copy.deepcopy(balancer),
                ) for i in range(num_codebooks)
            ]
        )

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
          prompt: Audio tokens as prompt, of shape (N, T, num_codebooks).

        Returns:
          Embedding tensor, of shape (N, T, embed_dim).
        """
        assert prompt.shape[-1] == self.num_codebooks, (
            prompt.shape, self.num_codebooks
        )

        embedding_out_list = []
        for i in range(self.num_codebooks):
            embedding_out = self.embeddings[i](prompt[:, :, i])
            embedding_out_list.append(embedding_out)

        # sum over different codebooks
        # (N, T, D)
        embedding_out = torch.stack(embedding_out_list, dim=0).sum(dim=0)

        return embedding_out


class DecoderEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1025,
        blank_id: int = 0,
        num_codebooks: int = 4,
        embed_dim: int = 512,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          blank_id:
            The ID of the blank symbol.
          num_codebooks:
            Number of codebooks used to encode audio.
          embed_dim:
            Dimension of the input embedding.
        """
        super().__init__()

        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks

        embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=blank_id,
        )
        balancer = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )
        self.embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    copy.deepcopy(embedding),
                    copy.deepcopy(balancer),
                ) for i in range(num_codebooks)
            ]
        )

    def forward(
        self,
        y: torch.Tensor,
        codebook_index: int,
        pre_ys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U) with BOS prepended.
          codebook_index:
            Index of codebook we want to predict, should be in range of
            [0, num_codebooks - 1].
          pre_ys:
            The previous codebooks of shape (N, U), with EOS prepended.

        Returns:
          Embedding tensor, of shape (N, T, embed_dim).
        """
        assert 0 <= codebook_index < self.num_codebooks, (
            codebook_index,
            self.num_codebooks,
        )

        embedding_out = self.embeddings[codebook_index](y)

        if codebook_index > 0:
            assert pre_ys is not None
            assert pre_ys.shape[-1] == codebook_index, (pre_ys.shape, codebook_index)

            embedding_out_list = [embedding_out]
            for i in range(codebook_index):
                embedding_out = self.embeddings[i](pre_ys[:, :, i])
                embedding_out_list.append(embedding_out)

            # sum over different codebooks
            embedding_out = torch.stack(embedding_out_list, dim=0).sum(dim=0)

        return embedding_out
