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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaling import Balancer


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_dim: int = 512,
        rnn_hidden_dim: int = 1024,
        num_layers: int = 1,
        rnn_dropout: float = 0.0,
    ):
        """
        Args:
          decoder_dim:
            Dimension of the decoder output.
          rnn_hidden_dim:
            Hidden dimension of LSTM layers.
          num_layers:
            Number of LSTM layers.
          rnn_dropout:
            Dropout for LSTM layers.
        """
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=decoder_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        self.out_proj = nn.Linear(rnn_hidden_dim, decoder_dim)
        self.out_balancer = Balancer(
            decoder_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          x:
            Embedding tensor of shape (N, U, D).
          states:
            A tuple of two tensors containing the states information of
            LSTM layers in this decoder.

        Returns:
          Return a tuple containing:
            - rnn_output, a tensor of shape (N, U, C)
            - (h, c), containing the state information for LSTM layers.
              Both are of shape (num_layers, N, C)
        """
        rnn_out, (h, c) = self.rnn(x, states)

        out = self.out_proj(rnn_out)
        out = nn.functional.relu(out)
        out = self.out_balancer(out)

        return out, (h, c)
