#!/usr/bin/env python3
# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey,
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


import copy
import math
import warnings
from typing import List, Optional, Tuple, Union
import logging
import torch
import random

# from encoder_interface import EncoderInterface
from scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    clip_grad,
    convert_num_channels,
)
from torch import Tensor, nn


try:
    import os
    import sys

    kernel_path = os.path.abspath(os.path.join(".."))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import (
        WindowProcess,
        WindowProcessReverse,
    )

except:  # noqa
    WindowProcess = None
    WindowProcessReverse = None
    print(
        "[Warning] Fused window process have not been installed. Please refer to get_started.md for installation."
    )


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LearnedDownsamplingModule(nn.Module):
    """
    Module that allows you to choose which frames to keep for transformer-type
    modules.  Effectively downsampling, but not necessarily "evenly"- you just
    keep some proportion of frames determined by the embedding.

    Args:
      embed_dim: embedding dimension
      downsampling_factor:  factor to downsample by, e.g. 2 or 4.  There is no
         fundamental reason why this has to be an integer, but we make it so
         anyway.
    """
    def __init__(self,
                 embed_dim: int,
                 downsampling_factor: int):
        assert downsampling_factor > 1

        super().__init__()

        self.to_scores = nn.Linear(embed_dim, 1, bias=False)
        self.to_scores.lr_scale = 0.5
        # score_balancer is just to keep the magnitudes of the scores in
        # a fixed range and keep them balanced around zero, to stop
        # these drifting around.
        # largish range used to keep grads relatively small and avoid overflow in grads.
        self.score_balancer = Balancer(1, channel_dim=-1,
                                       min_positive=1/(2*downsampling_factor),
                                       max_positive=0.6,
                                       min_abs=1.0,
                                       max_abs=4.0)

        # below are for diagnostics.
        self.copy_weights1 = nn.Identity()
        self.copy_weights2 = nn.Identity()

        self.downsampling_factor = downsampling_factor


    def forward(self,
                x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
          x: a Tensor of shape (seq_len, batch_size, embed_dim)

        Returns: (frame_indexes, weights, kept)

         frame_indexes: a Tensor of integer type, of shape (batch_size, reduced_seq_len)
              where reduced_seq_len = (seq_len + d - 1) // d.  It contains elements
              0 <= frame_indees < seq_len, in sorted (increasing) order

            weights: a Tensor of shape (batch_size, reduced_seq_len),
                 corresponding to the kept frames; these will be between 0 and 1, but
                 mostly exactly 1.
        """
        (seq_len, batch_size, _) = x.shape
        scores = self.to_scores(x)  # (seq_len, batch_size, 1)
        scores = self.score_balancer(scores)

        scores = scores.squeeze(-1).t()  # (batch_size, seq_len)

        # sscores, indexes: (batch_size, seq_len)
        sscores, indexes = scores.sort(dim=-1, descending=True)


        weights = sscores.clamp(min=0.0, max=1.0)
        weights = self.copy_weights1(weights)

        if self.training:
            d = self.downsampling_factor
            seq_len_reduced = (seq_len + d - 1) // d

            weights_discarded = weights[:, seq_len_reduced:2*seq_len_reduced]
            missing = seq_len_reduced - weights_discarded.shape[1]
            if missing != 0:
                weights_discarded = torch.cat((weights_discarded,
                                               torch.zeros(batch_size, missing,
                                                           device=weights.device,
                                                           dtype=weights.dtype)),
                                              dim=1)

            if random.random() < 0.01 or __name__ == '__main__':
                logging.info(f"mean weight={weights.mean()}, mean-abs-scores={scores.abs().mean()} positive-scores={(scores>0).to(torch.float32).mean()}, discarded-weights={weights_discarded.mean()}, seq_len={seq_len}, seq_len_reduced={seq_len_reduced}")


            if random.random() < 0.5:
                # flipping it half the time increases the randomness, so gives an extra incentive
                # to avoid nonzero weights in the discarded half
                weights_discarded = weights_discarded.flip(dims=(1,))

            weights = weights[:, :seq_len_reduced] - weights_discarded
        else:
            # test mode.  because the sequence might be short, we keep all nonzero scores;
            # and there is no need for any penalty.

            # need to work out seq_len_reduced.
            seq_len_reduced = max(1,
                                  (weights > 0.0).to(torch.int32).sum(dim=-1).max().item())
            if random.random() < 0.02:
                logging.info(f"seq_len={seq_len}, seq_len_reduced={seq_len_reduced}")
            weights = weights[:, :seq_len_reduced]

        indexes = indexes[:, :seq_len_reduced]


        weights = self.copy_weights2(weights)

        # re-sort the indexes we kept, on index value, so that
        # masking for causal models will be in the correct order.
        # (actually this may not really matter, TODO: see whether we
        # can remove this??)
        indexes, reorder = indexes.sort(dim=-1)
        weights = torch.gather(weights, dim=-1, index=reorder)

        x_downsampled = self.downsample(x, indexes)
        return indexes, weights, x_downsampled


    def downsample(self, x: Tensor, indexes: Tensor) -> Tensor:
        """
        Downsamples x via indexing with the indexes obtained from the
        forward() function.

        Args:
           x: tensor of shape (seq_len, batch_size, num_channels)
         indexes: integer indexes of shape (batch_size, seq_len_reduced), with elements
                 0 <= indexes < seq_len.
        Returns:
           x_downsampled, of shape (seq_len_reduced, batch_size, num_channels)
        """
        indexes_expanded = indexes.t().unsqueeze(-1).expand(-1, -1, x.shape[-1])
        # indexe_expanded: (seq_len_reduced, batch_size, num_channels)
        ans = torch.gather(x, dim=0, index=indexes_expanded)

        if __name__ == '__main__':
            # temp, for testing
            x_reconstructed = self.upsample(x, ans, indexes)
            assert torch.allclose(x, x_reconstructed)

        return ans


    def downsample_pos_emb(self, pos_emb: Tensor, indexes: Tensor) -> Tensor:
        """
        Downsample positional embedding tensor with the provided indexes.
        Args:
          pos_emb: (batch_size, seq_len, seq_len, pos_dim)
                interpreted as (batch_size, tgt_seq_len, src_seq_len, pos_dim).
          indexes: (batch_size, seq_len_reduced), containing integer elements
                  0 <= indexes < seq_len.
        Returns:
          downsampled_pos_len: (batch_size, seq_len_reduced, seq_len_reduced, pos_dim)
        """

        (batch_size, seq_len_reduced) = indexes.shape
        (_, _, seq_len, pos_dim) = pos_emb.shape

        tgt_indexes = indexes.reshape(batch_size, seq_len_reduced, 1, 1).expand(
            batch_size, seq_len_reduced, seq_len, pos_dim)

        pos_emb = torch.gather(pos_emb, dim=1, index=tgt_indexes)
        # now pos_emb: (batch_size, seq_len_reduced, seq_len, pos_dim)

        src_indexes = indexes.reshape(batch_size, 1, seq_len_reduced, 1).expand(
            batch_size, seq_len_reduced, seq_len_reduced, pos_dim)

        pos_emb = torch.gather(pos_emb, dim=2, index=src_indexes)
        # now pos_emb: (batch_size, seq_len_reduced, seq_len_reduced, pos_dim)
        return pos_emb


    def downsample_attn_offset(self,
                               attn_offset: Tensor,
                               indexes: Tensor,
                               weights: Tensor,
                               eps: float = 1.0e-03) -> Tensor:
        """
        Downsamples attn_offset and also modifies it to account for the weights in `weights`.
        Args:
              attn_offset: a Tensor of shape (1 or batch_size, seq_len, seq_len), interpreted as
                          (1 or batch_size, tgt_seq_len, src_seq_len)
                 indexes: a Tensor of shape (batch_size, reduced_seq_len) containing elements
                        0 <= indexes < seq_len.
                   weights: a Tensor of shape (batch_size, reduced_seq_len) containing weights
                          between 0 and 1; most will be 1.
        Returns:
              attn_offset_downsampled, a Tensor of shape (batch_size, reduced_seq_len, reduced_seq_len)
        """
        (batch_size, seq_len_reduced) = indexes.shape
        seq_len = attn_offset.shape[-1]
        assert len(attn_offset.shape) == 3  # (1, seq_len, seq_len) or (batch_size, seq_len, seq_len)
        attn_offset = attn_offset.expand(batch_size, seq_len, seq_len)

        if torch.is_autocast_enabled():
            # it's possible to get large gradients at this point; clip these at
            # this point to reduce the extent to which it has to reduce the
            # grad_scale.
            weights = clip_grad(weights, 5000.0)

        attn_offset = attn_offset.gather(dim=1, index=indexes.unsqueeze(-1).expand(
            batch_size, seq_len_reduced, seq_len))
        attn_offset = attn_offset.gather(dim=2, index=indexes.unsqueeze(1).expand(
            batch_size, seq_len_reduced, seq_len_reduced))
        # unsqueeze at position 1 so the extra cost relates to the source position.
        attn_offset = attn_offset + (weights + eps).log().unsqueeze(1)

        return attn_offset


    @staticmethod
    def upsample(x_orig: Tensor, x: Tensor, indexes: Tensor,
                 weights: Optional[Tensor] = None) -> Tensor:
        """
        Upsamples, reversing the downsample() operation and filling in
        any not-chosen frames with their original value before downsampling
        (or with whatever x_orig contains).

        Args:
            x_orig: (seq_len, batch_size, num_channels)
            x: (seq_len_reduced, batch_size, num_channels)
          indexes: (batch_size, seq_len_reduced), contains original frame indexes
          weights: optional tensor

        Downsamples x via indexing with the indexes obtained from the
        forward() function.

        Args:
            x: tensor of shape (seq_len, batch_size, indexes)
         weights: a tensor of shape (batch_size, seq_len_reduced) containing weights between
             0 and 1, where 1 means fully use this x value and 0 means use x_orig
         indexes: integer indexes of shape (batch_size, seq_len_reduced), with elements
                 0 <= indexes < seq_len.
        """
        (seq_len, batch_size, num_channels) = x_orig.shape

        x_weight = 1.0 if weights is None else weights.t().unsqueeze(-1)
        # x_weight: (seq_len_reduced, batch_size, 1) if a tensor

        orig_x_weight = torch.ones(batch_size, seq_len,
                                   device=x.device, dtype=x.dtype)
        if weights is None:
            orig_x_weight.scatter_(dim=1, index=indexes, value=0.)
        else:
            orig_x_weight.scatter_(dim=1, index=indexes,
                                   src=(1. - weights).to(x.dtype))

        indexes = indexes.t().unsqueeze(-1).expand(-1, batch_size, num_channels)
        # indexes now: (seq_len_reduced, batch_size, num_channels)

        ans = torch.zeros_like(x_orig)

        ans.scatter_(dim=0, index=indexes, src=(x * x_weight).to(dtype=ans.dtype))

        # add in x_orig in the frames that were not originally kept.
        return ans + x_orig * orig_x_weight.t().unsqueeze(-1)


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding;
    in this version, the positions for each frame are passed in (in order to support


    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
            pos_dim: dimension of the projected positional encoding, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
      pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                   any given call to forward(), in training time.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_dim: int,
            dropout: float = 0.0,
            pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                          (4000.0, 0.0))
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_dim = pos_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(embed_dim, in_proj_dim, bias=True,
                                    initial_scale=query_head_dim**-0.25)

        self.whiten_keys = Whiten(num_groups=num_heads,
                                  whitening_limit=_whitening_schedule(3.0),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.025)

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be suffixient to fix the problem.
        self.balance_keys = Balancer(key_head_dim * num_heads,
                                     channel_dim=-1,
                                     min_positive=0.4,
                                     max_positive=0.6,
                                     min_abs=0.0,
                                     max_abs=100.0,
                                     prob=0.025)


        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()


    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        attn_offset: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 2, pos_dim)

         attn_offset:  a Tensor of shape broadcasting with (batch_size, seq_len, seq_len),
             interpreted as (batch_size, tgt_seq_len, src_seq_len), if provided this
             contains values (probably <= 0) to be added to the logprobs of the attention;
             this may combine the log of 'weights' of ChooseDownsamplingModule with
             any attn_mask that enforces causality.
         pos_emb: a Tensor of shape broadcasting with (batch_size, seq_len, seq_len, pos_dim)
             (e.g. pos_dim=4), encoding relative positions.

        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_dim = self.pos_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        q = x[...,0:query_dim]
        k = x[...,query_dim:2*query_dim]
        # p is the position-encoding query
        p = x[...,2*query_dim:]
        assert p.shape[-1] == num_heads * pos_dim


        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.


        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        q = q.permute(2, 1, 0, 3)  # (head, batch, tgt_seq_len, query_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, src_seq_len)

        # attn_scores: (num_heads, batch_size, tgt_seq_len, src_esq_len)
        attn_scores = torch.matmul(q, k)

        if not self.training or random.random() >= float(self.pos_emb_skip_rate):
            #                   pos_emb: (batch_size, tgt_seq_len, src_seq_len, pos_dim)
            p = p.permute(1, 0, 3, 2)  # (batch_size, tgt_seq_len, pos_dim, num_heads)

            pos_scores = torch.matmul(pos_emb, p)
            # pos_scores: (batch_size, tgt_seq_len, src_seq_len, num_heads)
            pos_scores = pos_scores.permute(3, 0, 1, 2)
            # pos_scores: (num_heads, batch_size, tgt_seq_len, src_seq_len)
            attn_scores = attn_scores + pos_scores

        if self.training and random.random() < 0.1:
            # This is away of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 25.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(attn_scores,
                                                 limit=25.0,
                                                 penalty=1.0e-04,
                                                 name=self.name)

        # attn_offset includes key-padding mask and attention-mask, plus any weights
        # from the subsampling.
        attn_scores = attn_scores + attn_offset

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if random.random() < 0.001:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights


    def _print_attn_entropy(
            self,
            attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(
                    dim=-1).mean(dim=(1,2))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")


class Attention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim_in: the input embedding dimension
          embed_dim_out: the output embedding dimension (normally the same as input)
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """
    def __init__(
            self,
            embed_dim_in: int,
            embed_dim_out: int,
            num_heads: int,
            value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim_in,
                                 num_heads * value_head_dim,
                                 bias=False)

        self.out_proj = ScaledLinear(num_heads * value_head_dim,
                                     embed_dim_out, bias=False,
                                     initial_scale=0.05)

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)


    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, query_len, key_len),
          Expect attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (num_heads, batch_size, query_len, key_len) = attn_weights.shape

        x = self.in_proj(x)     #  (key_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(key_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, key_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, query_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(
            query_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (query_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        return x


class FeedforwardModule(nn.Module):
    """Feedforward module in Subformer model.
    """
    def __init__(self,
                 embed_dim: int,
                 feedforward_dim: int,
                 dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(feedforward_dim,
                                        channel_dim=-1,
                                        min_positive=0.3,
                                        max_positive=1.0,
                                        min_abs=0.75,
                                        max_abs=5.0)

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(feedforward_dim, embed_dim,
                                                   activation='SwooshL',
                                                   dropout_p=dropout,
                                                   dropout_shared_dim=0, bias=True,
                                                   initial_scale=0.1)

        self.out_whiten =  Whiten(num_groups=1,
                                  whitening_limit=_whitening_schedule(7.5),
                                  prob=(0.025, 0.25),
                                  grad_scale=0.01)

    def forward(self,
                x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
            self,
            channels: int,
            hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(hidden_channels, channels,
                                     bias=True,
                                     initial_scale=0.05)



        self.whiten1 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)

        self.whiten2 = Whiten(num_groups=1,
                              whitening_limit=_whitening_schedule(5.0, ratio=3.0),
                              prob=(0.025, 0.25),
                              grad_scale=0.01)


    def forward(self,
                x: Tensor,
                attn_weights: Tensor,
    ) -> Tensor:
        """.
        Args:
           x: a Tensor of shape (seq_len, batch_size, num_channels)
attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
        Returns:
           a Tensor with the same shape as x
        """
        num_channels = x.shape[-1]
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=-1)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)


        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x


class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """
    def __init__(
            self,
            embed_dim: int,
            skip_rate: FloatLike = 0.0,
            straight_through_rate: FloatLike = 0.0,
            scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
            scale_max: FloatLike = 1.0):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def embed_dim(self):
        return self.bypass_scale.numel()

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.scale_min),
                                    max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))

            return ans

    def forward(self,
                src_orig: Tensor,
                src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig)  * bypass_scale


class SubformerEncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = SubformerEncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            query_head_dim: int,
            value_head_dim: int,
            pos_dim: int,
            feedforward_dim: int,
            dropout: FloatLike = 0.1,
            causal: bool = False,
            attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
            ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.02), default=0),
    ) -> None:
        super(SubformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(embed_dim, skip_rate=bypass_skip_rate,
                                   straight_through_rate=0.025)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0.025)


        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, num_heads=num_heads,
            query_head_dim=query_head_dim, pos_dim=pos_dim,
            dropout=0.0,
        )


        self.self_attn1 = Attention(embed_dim, embed_dim, num_heads,
                                    value_head_dim)

        self.self_attn2 = Attention(embed_dim, embed_dim, num_heads,
                                    value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 3) // 4,
                                               dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim,
                                               feedforward_dim,
                                               dropout)

        self.feed_forward3 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 5) // 4,
                                               dropout)

        self.nonlin_attention = NonlinAttention(embed_dim,
                                                hidden_channels=3 * embed_dim // 4)


        #self.attention_squeeze = AttentionSqueeze(embed_dim, embed_dim // 2)

        self.norm = BiasNorm(embed_dim)

        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.balancer1 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.2, max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.3, max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(4.0, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

        self.balancer2 = Balancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.1, max_abs=4.0,
        )

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask


    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask


    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        attn_offset: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
         pos_emb: (batch_size, seq_len, seq_len, pos_dim), with e.g. pos_dim=4: relatie positional
               embedding tensor.
       feature_mask: something that broadcasts with src, that we'll multiply `src`
              by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
        attn_offset: the attention offset, of shape broadcasting with (batch_size, seq_len, seq_len),
                interpreted as (batch_size, tgt_seq_len, src_seq_len).  -inf for masked position.
     src_key_padding_mask: the mask for padding, of shape (batch_size, seq_len); True means
             masked position.  May be None.

        Returns:
           A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_offset=attn_offset,
        )

        if memory is not None and hasattr(self, 'attn_weights'):
            src_attn_weights = self.attn_weights(memory, src, memory_key_padding_mask)

        src = src + self.feed_forward1(src)

        attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        if True:
            selected_attn_weights = attn_weights[0:2]
            if random.random() < float(self.const_attention_rate):
                # Make attention weights constant.  The intention is to
                # encourage these modules to do something similar to an
                # averaging-over-time operation.
                # only need the mask, can just use the 1st one and expand later
                selected_attn_weights = selected_attn_weights[0:1]
                selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
                selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))
                selected_attn_weights = selected_attn_weights.expand(2, -1, -1, -1)


        na = self.balancer_na(self.nonlin_attention(src,
                                                    selected_attn_weights[0:1]))

        src = src + (na if attn_dropout_mask is None else na * attn_dropout_mask)

        self_attn = self.self_attn1(
            src, attn_weights)

        src = src + (self_attn if attn_dropout_mask is None else self_attn * attn_dropout_mask)

        if memory is not None and hasattr(self, 'attn_weights'):
            src = src + self.sequence_dropout(self.src_attn1(memory, src_attn_weights),
                                              attention_skip_rate)

        src = src + self.sequence_dropout(self.balancer_ff2(self.feed_forward2(src)),
                                          float(self.ff2_skip_rate))

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(
            src, attn_weights)

        src = src + (self_attn if attn_dropout_mask is None else self_attn * attn_dropout_mask)

        if memory is not None and hasattr(self, 'attn_weights'):
            src = src + self.sequence_dropout(self.src_attn2(memory, src_attn_weights),
                                              attention_skip_rate)

        src = src + self.sequence_dropout(self.balancer_ff3(self.feed_forward3(src)),
                                          float(self.ff3_skip_rate))

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works done by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish invidual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Temporary embedding dimension used inside this module
         pos_dim: Smaller positional-encoding dim used after a projecction.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
        pos_dim: dimension at the output of this module.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        dropout_rate: FloatLike,
        height: int,
        width: int,
        length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.pos_dim = pos_dim
        self.dropout = Dropout2(dropout_rate)
        assert length_factor >= 1.0
        self.length_factor = length_factor
        self.height = height
        self.width = width

        # linear transformation for positional encoding.
        self.linear_pos_row = ScaledLinear(
            embed_dim, pos_dim, bias=False, initial_scale=0.05
        )
        self.linear_pos_col = ScaledLinear(
            embed_dim, pos_dim, bias=False, initial_scale=0.05
        )

        self.pe_row = self.generate_pe(length=height)
        self.pe_col = self.generate_pe(length=width)

    def generate_pe(self, length: int) -> None:
        """Generate the positional encodings with given length."""
        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(length - 1), length).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = (
            compression_length
            * x.sign()
            * ((x.abs() + compression_length).log() - math.log(compression_length))
        )

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan()  # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.
        # pe: (2 * length - 1, embed_dim)
        return pe

    def forward(self, x: torch.Tensor) -> Tensor:
        """Create positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch_size, height, width, num_channels_in)

        Returns:
            positional embedding, of shape (height*width, heigh*width, pos_dim*2).
        """
        if self.pe_row.dtype != x.dtype or str(self.pe_row.device) != str(x.device):
            self.pe_row = self.pe_row.to(dtype=x.dtype, device=x.device)

        if self.pe_col.dtype != x.dtype or str(self.pe_col.device) != str(x.device):
            self.pe_col = self.pe_col.to(dtype=x.dtype, device=x.device)

        H = self.height
        W = self.width

        # (1, 2 * H - 1, pos_dim)
        row = self.linear_pos_row(self.dropout(self.pe_row.unsqueeze(0)))
        # (1, 2 * W - 1, pos_dim)
        col = self.linear_pos_col(self.dropout(self.pe_col.unsqueeze(0)))

        # expand the '1' dimension to seq_len; this introduces a dimension that
        # 'does nothing', just creates copies, as a workaround for lack of torch support
        # for negative strides.
        row = row.expand(H, 2 * H - 1, self.pos_dim).contiguous()
        (useless_stride, seq_stride, channel_stride) = row.stride()
        row = row.as_strided(
            (H, H, self.pos_dim),
            (useless_stride - seq_stride, seq_stride, channel_stride),
            storage_offset=seq_stride * (H - 1),
        )  # (H, H, pos_dim)

        # expand the '1' dimension to seq_len; this introduces a dimension that
        # 'does nothing', just creates copies, as a workaround for lack of torch support
        # for negative strides.
        col = col.expand(W, 2 * W - 1, self.pos_dim).contiguous()
        (useless_stride, seq_stride, channel_stride) = col.stride()
        col = col.as_strided(
            (W, W, self.pos_dim),
            (useless_stride - seq_stride, seq_stride, channel_stride),
            storage_offset=seq_stride * (W - 1),
        )  # (W, W, pos_dim)

        row = row[:, None, :, None, :].expand(H, W, H, W, -1)
        col = col[None, :, None, :, :].expand(H, W, H, W, -1)
        pos_emb = torch.stack([row, col], dim=-1)
        pos_emb = pos_emb.view(H * W, H * W, self.pos_dim * 2)

        return pos_emb


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        window_size (int): Window size. Default: 7
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        downsampling_factor (Tuple[int]): downsampling factor for DownsampledSubformerEncoderLayer
           in each stack
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
           Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.
        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        window_size: int = 7,
        num_encoder_layers: Tuple[int, ...] = (2, 2, 6, 2),
        downsampling_factor: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        query_head_dim: Tuple[int, ...] = (24,),
        value_head_dim: Tuple[int, ...] = (12,),
        pos_dim: int = 4,
        # feedforward_dim: Tuple[int, ...] = (1536,),
        mlp_ratio: float = 4.0,
        dropout: Optional[FloatLike] = None,  # see code below for default
        warmup_batches: float = 4000.0,
        fused_window_process=False,
    ):
        super(SwinTransformer, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        num_encoders = len(num_encoder_layers)

        def _to_tuple(x):
            """ Converts a single int or a 1-tuple of an int to a tuple with the same length
            as num_encoders"""
            assert isinstance(x, tuple)
            if len(x) == 1:
                x = x * num_encoders
            else:
                assert len(x) == num_encoders
            return x

        downsampling_factor = _to_tuple(downsampling_factor)
        query_head_dim = _to_tuple(query_head_dim)
        value_head_dim = _to_tuple(value_head_dim)
        num_heads = _to_tuple(num_heads)
        # feedforward_dim = _to_tuple(feedforward_dim)

        self.num_features = int(embed_dim * 2 ** (num_encoders - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        patches_resolution = self.patch_embed.patches_resolution

        assert pos_dim % 2 == 0, pos_dim
        self.pos_enc = CompactRelPositionalEncoding(
            embed_dim=64,
            pos_dim=pos_dim // 2,
            dropout_rate=0.15,
            height=window_size,
            width=window_size,
        )

        self.encoders = nn.ModuleList()
        for i in range(num_encoders):
            cur_embed_dim = int(embed_dim * 2 ** i)

            subformer_encoder_layer = SubformerEncoderLayer(
                embed_dim=cur_embed_dim,
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=int(cur_embed_dim * mlp_ratio),
                dropout=dropout,
            )
            ds = downsampling_factor[i]
            if ds != 1:
            # if False:
                subformer_encoder_layer = DownsampledSubformerEncoderLayer(
                    subformer_encoder_layer,
                    input_num_channels=cur_embed_dim,
                    downsampling_factor=ds,
                )

            encoder = SwinEncoder(
                subformer_encoder_layer,
                input_resolution=(patches_resolution[0] // (2 ** i),
                                  patches_resolution[1] // (2 ** i)),
                window_size=window_size,
                num_layers=num_encoder_layers[i],
                embed_dim=cur_embed_dim,
                downsample=True if (i < num_encoders - 1) else False,
                fused_window_process=fused_window_process,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
            )
            self.encoders.append(encoder)

        self.norm = BiasNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

    def forward_features(self, x: Tensor):
        x = self.patch_embed(x)  # (B, H*W, C)

        pos_emb = self.pos_enc(x)

        for encoder in self.encoders:
            x = encoder(x, pos_emb=pos_emb)

        x = self.norm(x)  # B H*W C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: Tensor):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class DownsampledSubformerEncoderLayer(nn.Module):
    """
    Args:
        encoder_layer: Instance of SubformerEncoderLayer class.
        input_num_channels: Embedding dimension.
        downsampling_factor: Downsampling fatcor.
    """
    def __init__(
        self,
        encoder_layer,
        input_num_channels: int,
        downsampling_factor: int = 2,  # TODO: change it to 4 in image?
    ):
        super(DownsampledSubformerEncoderLayer, self).__init__()
        if downsampling_factor != 1:
            self.downsampler = LearnedDownsamplingModule(
                input_num_channels, downsampling_factor
            )

        self.encoder_layer = encoder_layer

    def forward(self, src: Tensor, pos_emb: Tensor, attn_offset: Tensor):
        src_orig = src
        if hasattr(self, 'downsampler'):
            indexes, weights, src = self.downsampler(src)

            pos_emb = self.downsampler.downsample_pos_emb(pos_emb, indexes)

            attn_offset = self.downsampler.downsample_attn_offset(attn_offset,
                                                                  indexes,
                                                                  weights)


        src = self.encoder_layer(src, pos_emb=pos_emb, attn_offset=attn_offset)

        if hasattr(self, 'downsampler'):
            src = self.downsampler.upsample(src_orig, src, indexes, weights)

        return src


class SwinEncoderLayer(nn.Module):
    """
    It applies the DownsampledSubformerEncoderLayer or SubformerEncoderLayer on
    the windows or shifted wondows.
    """
    def __init__(
        self,
        subformer_encoder_layer: nn.Module,
        input_resolution: Tuple[int, int],
        window_size: int = 7,
        shift_size: int = 0,
        fused_window_process=False,
    ):
        super(SwinEncoderLayer, self).__init__()
        # DownsampledSubformerEncoderLayer or SubformerEncoderLayer
        self.subformer_encoder_layer = subformer_encoder_layer

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            # H, W = self.input_resolution
            # nW = (H // self.window_size) * (W // self.window_size)
            # L = self.window_size * self.window_size
            # attn_mask = torch.zeros(nW, L, L)
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x: Tensor, pos_emb: Tensor):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows = x_windows.permute(1, 0, 2)  # (window_size*window_size, nW*B, C)

        # Apply a subformer block on each windows
        L_, B_, _ = x_windows.size()
        if self.attn_mask is not None:
            attn_offsets = self.attn_mask.unsqueeze(0).expand(B, -1, L_, L_).reshape(B_, L_, L_).to(dtype=x.dtype)
        else:
            attn_offsets = torch.zeros(B_, L_, L_, dtype=x.dtype, device=x.device)
        # now attn_offsets: (nW*B, window_size*window_size, window_size*window_size)
        pos_emb = pos_emb.unsqueeze(0).expand(B_, L_, L_, -1)
        # now pos_emb: (nW*B, window_size*window_size, window_size*window_size, pos_dim)
        x_windows = self.subformer_encoder_layer(
            x_windows, attn_offset=attn_offsets, pos_emb=pos_emb
        )  # (window_size*window_size, nW*B, C)

        x_windows = x_windows.permute(1, 0, 2)  # (nW*B, window_size*window_size, C)
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(x_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(x_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(x_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x

        # now: x is (B, H, W, C)
        x = x.view(B, H * W, C)

        return x


class SwinEncoder(nn.Module):
    """
    An encoder stack which consists of multiple SwinEncoderLayers.
    The SwinEncoderLayers operates on windows and shifted windows alternatively.
    """
    def __init__(
        self,
        subformer_encoder_layer: nn.Module,
        input_resolution: Tuple[int, int],
        window_size: int,
        num_layers: int,
        embed_dim: int,
        warmup_begin: float,
        warmup_end: float,
        downsample: bool = True,
        fused_window_process: bool = False,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
    ):
        super(SwinEncoder, self).__init__()
        self.input_resolution = input_resolution

        self.layers = nn.ModuleList([
            SwinEncoderLayer(
                copy.deepcopy(subformer_encoder_layer),
                input_resolution=input_resolution,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                fused_window_process=fused_window_process,
            )
            for i in range(num_layers)
        ])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(input_resolution, embed_dim)

        assert 0 <= warmup_begin <= warmup_end

        delta = (1. / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            layer = self.layers[i].subformer_encoder_layer
            if hasattr(layer, "bypass"):
                # it is SubformerEncoderLayer
                bypass = layer.bypass
            else:
                # it is DownsampledSubformerEncoderLayer
                bypass = layer.encoder_layer.bypass
            bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end
            # print(self.layers[i].subformer_encoder_layer.encoder_layer.bypass.skip_rate.schedule.pairs)

    def forward(self, x: Tensor, pos_emb: Tensor):
        # x: (B, H*W, C)
        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        patches_resolution = [self.img_size[0] // self.patch_size[0],
                              self.img_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = BiasNorm(embed_dim, channel_dim=-1)
        self.out_balancer = Balancer(
            embed_dim, channel_dim=-1, min_positive=0.45, max_positive=0.55
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.out_balancer(x)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution: Tuple[int, int], embed_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)
        self.norm = BiasNorm(2 * embed_dim, channel_dim=-1)
        self.balancer = Balancer(
            2 * embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.2, max_abs=4.0,
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.balancer(x)
        x = self.norm(x)

        return x


def _test_rel_pos_enc():
    pos_enc = CompactRelPositionalEncoding(
        embed_dim=64, pos_dim=4, dropout_rate=0.1, height=7, width=6
    )
    x = torch.randn(2, 7 * 6, 384)
    pos_emb = pos_enc(x)
    pos_emb = pos_enc(x)
    print(pos_emb.shape)

    pos_emb = pos_emb.reshape(7, 6, 7, 6, 8)
    # a simple test
    assert torch.equal(pos_emb[1, 2, 3, 4, :], pos_emb[2, 3, 4, 5, :])
    assert torch.equal(pos_emb[4, 3, 2, 1, :], pos_emb[5, 4, 3, 2, :])


def _test_swin_transformer():
    model = SwinTransformer(num_classes=10, fused_window_process=False)
    model = model.cuda()

    for i in range(10):
        x = torch.randn(2, 3, 224, 224).cuda()
        y = model(x)
        print(y.shape)

        y.sum().backward()


if __name__ == "__main__":
    # _test_rel_pos_enc()
    _test_swin_transformer()
