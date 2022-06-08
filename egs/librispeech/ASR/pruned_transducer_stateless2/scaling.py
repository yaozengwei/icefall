# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey)
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


import collections
from itertools import repeat
from typing import Optional, Tuple
import logging

import random
import torch
import torch.nn as nn
from torch import Tensor


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)


class ActivationBalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        channel_dim: int,
        min_positive: float,  # e.g. 0.05
        max_positive: float,  # e.g. 0.95
        max_factor: float,  # e.g. 0.01
        min_abs: float,  # e.g. 0.2
        max_abs: float,  # e.g. 100.0
    ) -> Tensor:
        if x.requires_grad:
            if channel_dim < 0:
                channel_dim += x.ndim
            sum_dims = [d for d in range(x.ndim) if d != channel_dim]
            xgt0 = x > 0
            proportion_positive = torch.mean(
                xgt0.to(x.dtype), dim=sum_dims, keepdim=True
            )
            factor1 = (
                (min_positive - proportion_positive).relu()
                * (max_factor / min_positive)
                if min_positive != 0.0
                else 0.0
            )
            factor2 = (
                (proportion_positive - max_positive).relu()
                * (max_factor / (max_positive - 1.0))
                if max_positive != 1.0
                else 0.0
            )
            factor = factor1 + factor2
            if isinstance(factor, float):
                factor = torch.zeros_like(proportion_positive)

            mean_abs = torch.mean(x.abs(), dim=sum_dims, keepdim=True)
            below_threshold = mean_abs < min_abs
            above_threshold = mean_abs > max_abs

            ctx.save_for_backward(
                factor, xgt0, below_threshold, above_threshold
            )
            ctx.max_factor = max_factor
            ctx.sum_dims = sum_dims
        return x

    @staticmethod
    def backward(
        ctx, x_grad: Tensor
    ) -> Tuple[Tensor, None, None, None, None, None, None]:
        factor, xgt0, below_threshold, above_threshold = ctx.saved_tensors
        dtype = x_grad.dtype
        scale_factor = (
            (below_threshold.to(dtype) - above_threshold.to(dtype))
            * (xgt0.to(dtype) - 0.5)
            * (ctx.max_factor * 2.0)
        )

        neg_delta_grad = x_grad.abs() * (factor + scale_factor)
        return x_grad - neg_delta_grad, None, None, None, None, None, None


class BasicNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    So the idea is to introduce this large constant value as an explicit
    parameter, that takes the role of the "eps" in LayerNorm, so the network
    doesn't have to do this trick.  We make the "eps" learnable.

    Args:
       num_channels: the number of channels, e.g. 512.
      channel_dim: the axis/dimension corresponding to the channel,
        interprted as an offset from the input's ndim if negative.
        shis is NOT the num_channels; it should typically be one of
        {-2, -1, 0, 1, 2, 3}.
       eps: the initial "epsilon" that we add as ballast in:
             scale = ((input_vec**2).mean() + epsilon)**-0.5
          Note: our epsilon is actually large, but we keep the name
          to indicate the connection with conventional LayerNorm.
       learn_eps: if true, we learn epsilon; if false, we keep it
         at the initial value.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        eps: float = 0.25,
        learn_eps: bool = True,
    ) -> None:
        super(BasicNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps).log().detach())
        else:
            self.register_buffer("eps", torch.tensor(eps).log().detach())

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        scales = (
            torch.mean(x ** 2, dim=self.channel_dim, keepdim=True)
            + self.eps.exp()
        ) ** -0.5
        return x * scales


class ScaledLinear(nn.Linear):
    """
    A modified version of nn.Linear where the parameters are scaled before
    use, via:
         weight = self.weight * self.weight_scale.exp()
         bias = self.bias * self.bias_scale.exp()

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
        initial_speed: this affects how fast the parameter will
           learn near the start of training; you can set it to a
           value less than one if you suspect that a module
           is contributing to instability near the start of training.
           Nnote: regardless of the use of this option, it's best to
           use schedulers like Noam that have a warm-up period.
           Alternatively you can set it to more than 1 if you want it to
           initially train faster.   Must be greater than 0.
    """

    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs
    ):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)

        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in nn.Linear

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3 ** 0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        if self.bias is None or self.bias_scale is None:
            return None

        return self.bias * self.bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.linear(
            input, self.get_weight(), self.get_bias()
        )


class ScaledConv1d(nn.Conv1d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs
    ):
        super(ScaledConv1d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)
        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3 ** 0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        bias = self.bias
        bias_scale = self.bias_scale
        if bias is None or bias_scale is None:
            return None
        return bias * bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                self.get_weight(),
                self.get_bias(),
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input,
            self.get_weight(),
            self.get_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ScaledConv2d(nn.Conv2d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs
    ):
        super(ScaledConv2d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)
        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3 ** 0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in ** -0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        # see https://github.com/pytorch/pytorch/issues/24135
        bias = self.bias
        bias_scale = self.bias_scale
        if bias is None or bias_scale is None:
            return None
        return bias * bias_scale.exp()

    def _conv_forward(self, input, weight):
        F = torch.nn.functional
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                weight,
                self.get_bias(),
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.get_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.get_weight())


class ActivationBalancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.


    Args:
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           max_factor: the maximum factor by which we modify the derivatives for
              either the sign constraint or the magnitude constraint;
              e.g. with max_factor=0.02, the the derivatives would be multiplied by
              values in the range [0.98..1.02].
           min_abs:  the minimum average-absolute-value per channel, which
              we allow, before we start to modify the derivatives to prevent
              this.
           max_abs:  the maximum average-absolute-value per channel, which
               we allow, before we start to modify the derivatives to prevent
               this.
    """

    def __init__(
        self,
        channel_dim: int,
        min_positive: float = 0.05,
        max_positive: float = 0.95,
        max_factor: float = 0.01,
        min_abs: float = 0.2,
        max_abs: float = 100.0,
    ):
        super(ActivationBalancer, self).__init__()
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.max_factor = max_factor
        self.min_abs = min_abs
        self.max_abs = max_abs

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting():
            return x

        return ActivationBalancerFunction.apply(
            x,
            self.channel_dim,
            self.min_positive,
            self.max_positive,
            self.max_factor,
            self.min_abs,
            self.max_abs,
        )


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)
    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x = x.detach()
        s = torch.sigmoid(x - 1.0)
        y = x * s
        ctx.save_for_backward(s, y)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        s, y = ctx.saved_tensors
        return (y * (1 - s) + s) * y_grad


class DoubleSwish(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


class ScaledEmbedding(nn.Module):
    r"""This is a modified version of nn.Embedding that introduces a learnable scale
    on the parameters.  Note: due to how we initialize it, it's best used with
    schedulers like Noam that have a warmup period.

    It is a simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor.
                                 See Notes for more details regarding sparse gradients.

        initial_speed (float, optional):  This affects how fast the parameter will
           learn near the start of training; you can set it to a value less than
           one if you suspect that a module is contributing to instability near
           the start of training.  Nnote: regardless of the use of this option,
           it's best to use schedulers like Noam that have a warm-up period.
           Alternatively you can set it to more than 1 if you want it to
           initially train faster.  Must be greater than 0.


    Attributes:
        weight (Tensor): the learnable weights of the module of shape (num_embeddings, embedding_dim)
                         initialized from :math:`\mathcal{N}(0, 1)`

    Shape:
        - Input: :math:`(*)`, LongTensor of arbitrary shape containing the indices to extract
        - Output: :math:`(*, H)`, where `*` is the input shape and :math:`H=\text{embedding\_dim}`

    .. note::
        Keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's :class:`optim.SGD` (`CUDA` and `CPU`),
        :class:`optim.SparseAdam` (`CUDA` and `CPU`) and :class:`optim.Adagrad` (`CPU`)

    .. note::
        With :attr:`padding_idx` set, the embedding vector at
        :attr:`padding_idx` is initialized to all zeros. However, note that this
        vector can be modified afterwards, e.g., using a customized
        initialization method, and thus changing the vector used to pad the
        output. The gradient for this vector from :class:`~torch.nn.Embedding`
        is always zero.

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding = nn.Embedding(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        >>> embedding(input)
        tensor([[[-0.0251, -1.6902,  0.7172],
                 [-0.6431,  0.0748,  0.6969],
                 [ 1.4970,  1.3448, -0.9685],
                 [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                 [ 0.4362, -0.4004,  0.9400],
                 [-0.6431,  0.0748,  0.6969],
                 [ 0.9124, -2.3616,  1.1151]]])


        >>> # example with padding_idx
        >>> embedding = nn.Embedding(10, 3, padding_idx=0)
        >>> input = torch.LongTensor([[0,2,0,5]])
        >>> embedding(input)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.1535, -2.0309,  0.9315],
                 [ 0.0000,  0.0000,  0.0000],
                 [-0.1655,  0.9897,  0.0635]]])

    """
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        initial_speed: float = 1.0,
    ) -> None:
        super(ScaledEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

        self.scale = nn.Parameter(torch.zeros(()))  # see reset_parameters()
        self.sparse = sparse

        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters(initial_speed)

    def reset_parameters(self, initial_speed: float = 1.0) -> None:
        std = 0.1 / initial_speed
        nn.init.normal_(self.weight, std=std)
        nn.init.constant_(self.scale, torch.tensor(1.0 / std).log())

        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        scale = self.scale.exp()
        if input.numel() < self.num_embeddings:
            return (
                F.embedding(
                    input,
                    self.weight,
                    self.padding_idx,
                    None,
                    2.0,  # None, 2.0 relate to normalization
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                * scale
            )
        else:
            return F.embedding(
                input,
                self.weight * scale,
                self.padding_idx,
                None,
                2.0,  # None, 2.0 relates to normalization
                self.scale_grad_by_freq,
                self.sparse,
            )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}, scale={scale}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)


class GaussProjDrop(torch.nn.Module):
    """
    This has an effect similar to torch.nn.Dropout, but does not privilege the on-axis directions.
    The directions of dropout are fixed when the class is initialized, and are orthogonal.

    dropout_rate: the dropout probability (actually will define the number of zeroed-out directions)
     channel_dim: the axis corresponding to the channel, e.g. -1, 0, 1, 2.
    """
    def __init__(self,
                 num_channels: int,
                 dropout_rate: float = 0.1,
                 channel_dim: int = -1):
        super(GaussProjDrop, self).__init__()
        self.dropout_rate = dropout_rate
        # this formula for rand_scale was found empirically, trying to match the
        # statistics of dropout in terms of cross-correlation with the input, see
        # _test_gauss_proj_drop()
        self.rand_scale = (dropout_rate / (1-dropout_rate)) ** 0.5 # * (num_channels ** -0.5)

        self.channel_dim = channel_dim

        rand_mat = torch.randn(num_channels, num_channels)
        U, _, _ = rand_mat.svd()
        self.register_buffer('U', U)  # a random orthogonal square matrix.  will be a buffer.


    def _randperm_like(self, x: Tensor):
        """
        Returns random permutations of the integers [0,1,..x.shape[-1]-1],
        with the same shape as x.  All dimensions of x other than the last dimension
        will be treated as batch dimensions.

        Torch's randperm does not support a batch dimension, so we pseudo-randomly simulate it.

        For now, requires x.shape[-1] to be either a power of 2 or 3 times a power of 2, as
        we normally set channel dims.  This is required for some number theoretic stuff.
        """
        n = x.shape[-1]

        assert n & (n-1) == 0 or (n//3 & (n//3 - 1)) == 0

        b = x.numel() // n
        randint = random.randint(0, 1000)
        perm = torch.randperm(n, device=x.device)
        # ensure all elements of batch_rand are coprime to n; this will ensure
        # that multiplying the permutation by batch_rand and taking modulo
        # n leaves us with permutations.
        batch_rand = torch.arange(b, device=x.device) * (randint * 6) + 1
        batch_rand = batch_rand.unsqueeze(-1)
        ans = (perm * batch_rand) % n
        ans = ans.reshape(x.shape)
        return ans


    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        else:
            x = x.transpose(self.channel_dim, -1)  # (..., num_channels)
            x_bypass = x # will be used for "+ I"
            perm = self._randperm_like(x)
            x = torch.gather(x, -1, perm)
            # self.U will act like a different matrix for every row of x, because of the random
            # permutation.
            x = torch.matmul(x, self.U)
            x_next = torch.empty_like(x)
            # scatter_ uses perm in opposite way
            # from gather, inverting it.
            x_next.scatter_(-1, perm, x)
            x = (x_next * self.rand_scale + x_bypass)
            return x


class Decorrelate(torch.nn.Module):
    """
    This module is something similar to dropout; it is a random transformation that
    does nothing in eval mode.
    It is designed specifically to encourage the input data to be decorrelated, i.e.
    to have a diagonal covariance matrix (not necessarily unity).

    To save time, in training mode we only apply it on randomly selected minibatches.

    Args:
           num_channels:  The number of channels, e.g. 256.
             apply_prob:  The probability with which we apply this each time, in
                          training mode.  This is to save time (but of course it
                          will tend to make the effect weaker).
            dropout_rate: This number determines the scale of the random multiplicative
                          noise, in such a way that the self-correlation and cross-correlation
                          statistics match those dropout with the same `dropout_rate`
                          (assuming we applied the transform, e.g. if apply_prob == 1.0)
                          This number applies when the features are un-correlated.
        max_dropout_rate: This is an upper limit, for safety, on how aggressive the
                          randomization can be.
                    eps:  An epsilon used to prevent division by zero.
                    beta: A value 0 < beta < 1 that controls decay of covariance stats
   """
    def __init__(self,
                 num_channels: int,
                 apply_prob: float = 0.25,
                 dropout_rate: float = 0.1,
                 eps: float = 1.0e-04,
                 beta: float = 0.95,
                 channel_dim: int = -1):
        super(Decorrelate, self).__init__()
        self.apply_prob = apply_prob
        self.dropout_rate = dropout_rate
        self.channel_dim = channel_dim
        self.eps = eps
        self.beta = beta

        #rand_mat = torch.randn(num_channels, num_channels)
        #U, _, _ = rand_mat.svd()
        #self.register_buffer('U', U)  # a random orthogonal square matrix.  will be a buffer.

        self.register_buffer('T1', torch.eye(num_channels))
        self.register_buffer('rand_scales', torch.zeros(num_channels))
        self.register_buffer('nonrand_scales', torch.ones(num_channels))
        self.register_buffer('T2', torch.eye(num_channels))
        self.register_buffer('cov', torch.zeros(num_channels, num_channels))
        self.step = 0



    def _update_covar_stats(self, x: Tensor) -> None:
        """
        Args:
             x: Tensor of shape (*, num_channels)
         Updates covariance stats self.cov
        """
        x = x.detach()
        x = x.reshape(-1, x.shape[-1])
        x = x * (x.shape[0] ** -0.5)  # avoid overflow in half precision
        cov = torch.matmul(x.t(), x)
        self.cov.mul_(self.beta).add_(cov, alpha=(1-self.beta))
        self.step += 1

    def _update_transforms(self):

        norm_cov, inv_sqrt_diag = self._normalize_covar(self.cov)

        U, S, _ = norm_cov.svd()

        if random.random() < 0.1:
            logging.info(f"Decorrelate: max,min eig of normalized cov is: {S.max().item():.2e},{S.min().item():.2e}")

        # row indexes of U correspond to channels, column indexes correspond to
        # singular values: cov = U * diag(S) * U.t() where * is matmul.
        S_eps = S + self.eps
        S_sqrt = S_eps ** 0.5
        S_inv_sqrt = (S + self.eps) ** -0.5

        # Transform T1, which we'll incorporate as torch.matmul(x, self.T1), is:
        #  (i) multiply by inv_sqrt_diag which makes the covariance have
        #     a unit diagonal.
        #  (ii) multiply by U, which diagonalizes norm_cov (uncorrelated channels)
        #  (iii) divide by S_sqrt, which makes all dims have unit variance.
        self.T1[:] = (inv_sqrt_diag.unsqueeze(-1) * U / S_sqrt)

        # Transform T1, which we'll incorporate as torch.matmul(x, self.TT), is:
        #  (i) multiply by S_sqrt, which restors the variance of different dims,
        #  (ii) multiply by U, which diagonalizes norm_cov (uncorrelated channels)
        #  (iii) divide by inv_sqrt_diag which makes the covariance have its original
        #       diagonal values.
        self.T2[:] = (S_sqrt.unsqueeze(-1) * U.t() / inv_sqrt_diag)


        # OK, now get rand_scales, which are values between 0 and self.dropout_rate; it says
        # how much randomness will be in different eigenvalues of norm_cov.
        # Basically, we want more randomness in directions with eigenvalues more than one,
        # and none in those with eigenvalues less than one.
        rand_proportion = (S - 1.0).clamp(min=0.0, max=1.0) * self.dropout_rate

        # rand_proportion is viewed as representing a proportion of the covariance, since
        # the random and nonrandom components will not be correlated.
        self.rand_scales[:] = rand_proportion.sqrt()
        self.nonrand_scales[:] = (1.0 - rand_proportion).sqrt()


        if True:
            d = torch.matmul(self.T1, self.T2) - torch.eye(self.T1.shape[0],
                                                           device=self.T1.device,
                                                           dtype=self.T1.dtype)
            assert torch.all(d.abs() < 0.01)



    def _normalize_covar(self, cov: Tensor) -> Tensor:
        """
        Normlizes a covariance matrix so that its diagonal is 1, by multiplying by
        its diagonal**-0.5 on both sides.
        Args:
          cov: matrix to normalize
        Returns normalized_cov, inv_sqrt_diag
        """
        diag = cov.diag()
        inv_sqrt_diag = (diag + self.eps) ** -0.5
        cov = cov * (inv_sqrt_diag * inv_sqrt_diag.unsqueeze(-1))
        assert torch.all((cov.diag() - 1.0).abs() < 0.1) # TODO: remove
        return cov, inv_sqrt_diag


    def _randperm_like(self, x: Tensor):
        """
        Returns random permutations of the integers [0,1,..x.shape[-1]-1],
        with the same shape as x.  All dimensions of x other than the last dimension
        will be treated as batch dimensions.

        Torch's randperm does not support a batch dimension, so we pseudo-randomly simulate it.

        For now, requires x.shape[-1] to be either a power of 2 or 3 times a power of 2, as
        we normally set channel dims.  This is required for some number theoretic stuff.
        """
        n = x.shape[-1]

        assert n & (n-1) == 0 or (n//3 & (n//3 - 1)) == 0

        b = x.numel() // n
        randint = random.randint(0, 1000)
        perm = torch.randperm(n, device=x.device)
        # ensure all elements of batch_rand are coprime to n; this will ensure
        # that multiplying the permutation by batch_rand and taking modulo
        # n leaves us with permutations.
        batch_rand = torch.arange(b, device=x.device) * (randint * 6) + 1
        batch_rand = batch_rand.unsqueeze(-1)
        ans = (perm * batch_rand) % n
        ans = ans.reshape(x.shape)
        return ans


    def forward(self, x: Tensor) -> Tensor:
        if not self.training or random.random() > self.apply_prob:
            return x
        else:
            x = x.transpose(self.channel_dim, -1)  # (..., num_channels)

            with torch.cuda.amp.autocast(enabled=False):
                self._update_covar_stats(x)
                if self.step % 50 == 0 or __name__ == "__main__":
                    self._update_transforms()

            x = torch.matmul(x, self.T1)

            x_bypass = x

            if False:
                # This block, in effect, multiplies x by a random orthogonal matrix,
                # giving us random noise.
                perm = self._randperm_like(x)
                x = torch.gather(x, -1, perm)
                # self.U will act like a different matrix for every row of x,
                # because of the random permutation.
                x = torch.matmul(x, self.U)
                x_next = torch.empty_like(x)
                # scatter_ uses perm in opposite way
                # from gather, inverting it.
                x_next.scatter_(-1, perm, x)
                x = x_next

            mask = (torch.randn_like(x) > 0.5)
            x = x - (x * mask) * 2

            x = (x * self.rand_scales) + (x_bypass * self.nonrand_scales)

            x = torch.matmul(x, self.T2)
            x = x.transpose(self.channel_dim, -1)
            return x




def _test_activation_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        channel_dim=0,
        min_positive=0.05,
        max_positive=0.95,
        max_factor=0.2,
        min_abs=0.0,
    )

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_activation_balancer_sign: x = ", x)
    print("_test_activation_balancer_sign: y grad = ", y_grad)
    print("_test_activation_balancer_sign: x grad = ", x.grad)


def _test_activation_balancer_magnitude():
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N)) * magnitudes.unsqueeze(
        -1
    )
    x = x.detach()
    x.requires_grad = True
    m = ActivationBalancer(
        channel_dim=0,
        min_positive=0.0,
        max_positive=1.0,
        max_factor=0.2,
        min_abs=0.2,
        max_abs=0.8,
    )

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_activation_balancer_magnitude: x = ", x)
    print("_test_activation_balancer_magnitude: y grad = ", y_grad)
    print("_test_activation_balancer_magnitude: x grad = ", x.grad)


def _test_basic_norm():
    num_channels = 128
    m = BasicNorm(num_channels=num_channels, channel_dim=1)

    x = torch.randn(500, num_channels)

    y = m(x)

    assert y.shape == x.shape
    x_rms = (x ** 2).mean().sqrt()
    y_rms = (y ** 2).mean().sqrt()
    print("x rms = ", x_rms)
    print("y rms = ", y_rms)
    assert y_rms < x_rms
    assert y_rms > 0.5 * x_rms


def _test_double_swish_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 0.5
    x.requires_grad = True
    m = DoubleSwish()
    torch.autograd.gradcheck(m, x)


def _test_gauss_proj_drop():
    D = 384
    x = torch.randn(30000, D)


    for dropout_rate in [0.2, 0.1, 0.01, 0.05]:
        m1 = torch.nn.Dropout(dropout_rate)
        m2 = GaussProjDrop(D, dropout_rate)
        for mode in ['train', 'eval']:
            y1 = m1(x)
            y2 = m2(x)
            xmag = (x*x).mean()
            y1mag = (y1*y1).mean()
            cross1 = (x*y1).mean()
            y2mag = (y2*y2).mean()
            cross2 = (x*y2).mean()
            print(f"rate={dropout_rate}, mode={mode}, xmag = {xmag}, y1mag = {y1mag}, y2mag = {y2mag}, cross1={cross1}, cross2={cross2}")
            m1.eval()
            m2.eval()

def _test_decorrelate():
    logging.getLogger().setLevel(logging.INFO)
    D = 384
    x = torch.randn(30000, D)

    # give it a non-unit covariance.
    m = torch.randn(D, D)
    x = torch.matmul(x, m)


    for dropout_rate in [0.2, 0.1, 0.01, 0.05]:
        m1 = torch.nn.Dropout(dropout_rate)
        m2 = Decorrelate(D, apply_prob=1.0, dropout_rate=dropout_rate)
        for mode in ['train', 'eval']:
            y1 = m1(x)
            y2 = m2(x)
            xmag = (x*x).mean()
            y1mag = (y1*y1).mean()
            cross1 = (x*y1).mean()
            y2mag = (y2*y2).mean()
            cross2 = (x*y2).mean()
            print(f"rate={dropout_rate}, mode={mode}, xmag = {xmag}, y1mag = {y1mag}, y2mag = {y2mag}, cross1={cross1}, cross2={cross2}, ratio1={y1mag/cross1}, ratio2={y2mag/cross2}")
            m1.eval()
            m2.eval()


if __name__ == "__main__":
    _test_decorrelate()
    _test_gauss_proj_drop()
    _test_activation_balancer_sign()
    _test_activation_balancer_magnitude()
    _test_basic_norm()
    _test_double_swish_deriv()
