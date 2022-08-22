# Copyright    2021  Xiaomi Corp.        (authors: Zengwei)
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


from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from scaling import ScaledLinear


class RegressLossNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = ScaledLinear(input_dim, output_dim)
        self.kl_loss = nn.KLDivLoss(log_target=True, reduction="none")

    def forward(
        self,
        x: torch.Tensor,
        y: torch.tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        assert reduction in ("sum", "mean"), reduction

        prediction = self.linear(x)

        # mse loss
        # loss = ((prediction - y) ** 2).mean(dim=2)

        # Kullback-Leibler divergence loss
        # (N, T)
        loss = self.kl_loss(
            F.log_softmax(prediction, dim=2), F.log_softmax(y, dim=2)
        ).sum(dim=2)
        if mask is not None:
            assert loss.shape == mask.shape, (loss.shape, mask.shape)
            loss.masked_fill_(mask, 0)

        loss = loss.sum() if reduction == "sum" else loss.mean()
        return loss
