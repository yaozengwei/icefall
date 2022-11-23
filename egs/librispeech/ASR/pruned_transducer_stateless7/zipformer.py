#!/usr/bin/env python3
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
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
import itertools
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    MaxEig,
    DoubleSwish,
    TanSwish,
    ScaledConv1d,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
)
from torch import Tensor, nn

from icefall.utils import make_pad_mask
from icefall.dist import get_rank


class Zipformer(EncoderInterface):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.


        num_features (int): Number of input features, e.g. 40.
        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        encoder_unmasked_dim (int or Tuple[int]): unmasked dimension in each of
            the encoder stacks for purposes of per-frame dropout (recommend 256 for
            now).
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
    """

    def __init__(
            self,
            num_features: int,
            output_downsampling_factor: int = 2,
            downsampling_factor: Tuple[int] = (2, 4),
            encoder_dim: Union[int, Tuple[int]] = 384,
            num_encoder_layers: Union[int, Tuple[int]] = 4,
            encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
            query_head_dim: Union[int, Tuple[int]]  = 24,
            pos_head_dim: Union[int, Tuple[int]]  = 4,
            value_head_dim: Union[int, Tuple[int]] = 12,
            num_heads: Union[int, Tuple[int]] = 8,
            feedforward_dim: Union[int, Tuple[int]] = 1536,
            cnn_module_kernel: Union[int, Tuple[int]] = 31,
            pos_dim: int = 192,
            dropout: float = 0.1,
            warmup_batches: float = 4000.0,
    ) -> None:
        super(Zipformer, self).__init__()

        # this is not the probability of skipping a layer.  It is the probability of
        # dropping out the "skip module" which allows the model to skip groups of
        # encoder stacks; when it's dropped out like this, it means we are forced
        # to take the "direct" (non-bypass) path.
        self.layer_skip_dropout_prob = ScheduledFloat((0.0, 0.5),
                                                      (warmup_batches, 0.025))

        def _to_tuple(x):
            """ Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.num_features = num_features  # int
        self.output_downsampling_factor = output_downsampling_factor # int
        self.downsampling_factor = downsampling_factor # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim) # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(encoder_unmasked_dim) # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        query_head_dim = _to_tuple(query_head_dim)
        value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        cnn_module_kernel = _to_tuple(cnn_module_kernel)


        for u,d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, (T - 7) // 2, encoder_dims).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> (T - 7) // 2
        #   (2) embedding: num_features -> encoder_dims
        # In the normal configuration, we will downsample once more at the end
        # by a factor of 2, and most of the encoder stacks will run at a lower
        # sampling rate.
        self.encoder_embed = Conv2dSubsampling(num_features, encoder_dim[0],
                                               dropout=dropout)


        # each one will be ZipformerEncoder or DownsampledZipformerEncoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = ZipformerEncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = ZipformerEncoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1)
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformerEncoder(
                    encoder,
                    input_dim=encoder_dim[i-1] if i > 0 else encoder_dim[0],
                    output_dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                )
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # initializes self.skip_layers and self.skip_modules
        self._init_skip_modules()

        self.downsample_output = AttentionDownsample(encoder_dim[-1],
                                                     encoder_dim[-1],
                                                     downsample=output_downsampling_factor)


    def _init_skip_modules(self):
        """
        If self.downampling_factor = (1, 2, 4, 8, 4, 2), then at the input of layer
        indexed 4 (in zero indexing), with has subsapling_factor=4, we combine the output of
        layers 2 and 3; and at the input of layer indexed 5, which which has subsampling_factor=2,
        we combine the outputs of layers 1 and 5.
        """
        skip_layers = []
        skip_modules = []
        z = self.downsampling_factor
        for i in range(len(z)):
            if i <= 1 or z[i-1] <= z[i]:
                skip_layers.append(None)
                skip_modules.append(Identity())
            else:
                # TEMP
                for j in range(i-2, -1, -1):
                    if z[j] <= z[i] or j == 0:
                        # TEMP logging statement.
                        logging.info(f"At encoder stack {i}, which has downsampling_factor={z[i]}, we will "
                                     f"combine the outputs of layers {j} and {i-1}, with downsampling_factor={z[j]} and {z[i-1]}.")
                        skip_layers.append(j)
                        skip_modules.append(SimpleCombiner(self.encoder_dim[j],
                                                           self.encoder_dim[i-1],
                                                           min_weight=(0.0, 0.25)))
                        break
        self.skip_layers = skip_layers
        self.skip_modules = nn.ModuleList(skip_modules)

    def get_feature_masks(
            self,
            x: torch.Tensor) -> List[Union[float, Tensor]]:
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all enocder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoer dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (num_frames, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [ 1.0 ] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0

        max_downsampling_factor = max(self.downsampling_factor)

        num_frames_max = (num_frames0 + max_downsampling_factor - 1)

        feature_mask_dropout_prob = 0.15

        # frame_mask_max shape: (num_frames_max, batch_size, 1)
        frame_mask_max = (torch.rand(num_frames_max, batch_size, 1,
                                    device=x.device) >
                          feature_mask_dropout_prob).to(x.dtype)

        feature_masks = []
        for i in range(num_encoders):
            ds = self.downsampling_factor[i]
            upsample_factor = (max_downsampling_factor // ds)

            frame_mask = (frame_mask_max.unsqueeze(1).expand(num_frames_max, upsample_factor,
                                                            batch_size, 1)
                          .reshape(num_frames_max * upsample_factor, batch_size, 1))
            num_frames = (num_frames0 + ds - 1) // ds
            frame_mask = frame_mask[:num_frames]
            feature_mask = torch.ones(num_frames, batch_size, self.encoder_dim[i],
                                      dtype=x.dtype, device=x.device)
            u = self.encoder_unmasked_dim[i]
            feature_mask[:, :, u:] *= frame_mask
            feature_masks.append(feature_mask)

        return feature_masks


    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, encoder_dim[-1])
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lengths = (x_lens - 7) // 2
        assert x.size(0) == lengths.max().item()
        mask = make_pad_mask(lengths)

        outputs = []
        feature_masks = self.get_feature_masks(x)

        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            if self.skip_layers[i] is not None:
                # this how we implement U-net-like skipping of some series of
                # stacks.  The layer_skip_dropout_prob is to discourage it, especially
                # early in training, from completely ignoring the middle layers.
                if not (self.training and random.random() < float(self.layer_skip_dropout_prob)):
                    x = self.skip_modules[i](outputs[self.skip_layers[i]], x)
            x = module(x,
                       feature_mask=feature_masks[i],
                       src_key_padding_mask=None if mask is None else mask[...,::ds])
            outputs.append(x)

        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2
        lengths = (lengths + 1) // 2

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths


class ZipformerEncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """
    def __init__(
            self,
            embed_dim: int,
            pos_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_head_dim: int,
            value_head_dim: int,
            feedforward_dim: int,
            dropout: float = 0.1,
            cnn_module_kernel: int = 31,
            # layer_skip_rate will be overwritten to change warmup begin and end times.
            # treating batch_index == 0.0 specially is just to get scan_pessimistic_batches_for_oom()
            # to work correctly.
            layer_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.05), default=0),
            dynamic_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0),
            const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
            bypass_min: FloatLike = ScheduledFloat((0.0, 0.75), (20000.0, 0.25), default=0),
            bypass_max: FloatLike = 1.0,
    ) -> None:
        super(ZipformerEncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # probability of skipping the entire layer.
        self.layer_skip_rate = copy.deepcopy(layer_skip_rate)
        # skip probability for dynamic modules (meaning: anything but feedforward)
        self.dynamic_skip_rate = copy.deepcopy(dynamic_skip_rate)
        # min and max for self.bypass_scale, applied with probability 0.5 to avoid grads
        # ever becoming zero.
        self.bypass_min = copy.deepcopy(bypass_min)
        self.bypass_max = copy.deepcopy(bypass_max)
        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim, pos_dim=pos_dim, num_heads=num_heads,
            query_head_dim=query_head_dim, pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn = SelfAttention(embed_dim, num_heads,
                                        value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim,
                                               feedforward_dim,
                                               dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim,
                                               feedforward_dim,
                                               dropout)

        #self.conv_module1 = ConvolutionModule(embed_dim,
        #cnn_module_kernel)
        self.nonlin_attention_module = NonlinAttentionModule(embed_dim)


        self.conv_module = ConvolutionModule(embed_dim,
                                             cnn_module_kernel)


        self.attention_squeeze1 = AttentionSqueeze(embed_dim)
        self.attention_squeeze2 = AttentionSqueeze(embed_dim)

        self.norm_final = BasicNorm(embed_dim)

        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            embed_dim, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            max_abs=6.0,
        )
        self.whiten = Whiten(num_groups=1,
                             whitening_limit=5.0,
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

    def get_bypass_scale(self):
        if torch.jit.is_scripting() or not self.training:
            return self.bypass_scale
        else:
            return limit_param_value(self.bypass_scale,
                                     min=float(self.bypass_min),
                                     max=float(self.bypass_max))

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        batch_split: if not None, this layer will only be applied to

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, N is the batch size, E is the feature number
        """
        if self.training and random.random() < float(self.layer_skip_rate):
            # skip the layer
            return src

        src_orig = src

        # dropout rate for non-feedforward submodules
        dynamic_skip_rate = float(self.dynamic_skip_rate) if self.training else 0.0
        # multi-headed self-attention module
        use_self_attn = (random.random() >= dynamic_skip_rate)

        if torch.jit.is_scripting() or use_self_attn:
            # attn_weights: (num_heads, batch_size, seq_len, seq_len)
            attn_weights = self.self_attn_weights(
                src,
                pos_emb=pos_emb,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )

            first_attn_weights = attn_weights[0:3]
            if random.random() < float(self.const_attention_rate):
                # Make attention weights constant.  The intention is to
                # encourage these modules to do something similar to an
                # averaging-over-time operation.
                # only need the mask, can just use the 1st one and expand later
                first_attn_weights = first_attn_weights[0:1]
                first_attn_weights = (first_attn_weights > 0.0).to(first_attn_weights.dtype)
                first_attn_weights = first_attn_weights * (1.0 / first_attn_weights.sum(dim=-1, keepdim=True))
                first_attn_weights = first_attn_weights.expand(3, -1, -1, -1)

        if torch.jit.is_scripting() or use_self_attn:
            src = src + self.nonlin_attention_module(src,
                                                     first_attn_weights[0:1])

        src = src + self.feed_forward1(src)

        # pooling module
        if torch.jit.is_scripting() or use_self_attn:
            src = src + self.attention_squeeze1(src, first_attn_weights[1:2])

        if torch.jit.is_scripting() or use_self_attn:
            src = src + self.self_attn(
                src, attn_weights)

        if torch.jit.is_scripting() or random.random() >= dynamic_skip_rate:
            src = src + self.conv_module(src, src_key_padding_mask=src_key_padding_mask)

        src = src + self.feed_forward2(src)

        # pooling module
        if torch.jit.is_scripting() or use_self_attn:
            src = src + self.attention_squeeze2(src, first_attn_weights[2:3])

        src = self.norm_final(self.balancer(src))

        delta = src - src_orig

        src = src_orig + delta * self.get_bypass_scale()

        return self.whiten(src)


class ZipformerEncoder(nn.Module):
    r"""ZipformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the ZipformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
       pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = ZipformerEncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = ZipformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """
    def __init__(
            self,
            encoder_layer: nn.Module,
            num_layers: int,
            pos_dim: int,
            dropout: float,
            warmup_begin: float,
            warmup_end: float,
            initial_layerdrop_rate: float = 0.5,
            final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.15,
                                                        length_factor=3.0)

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end

        delta = (1. / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            # treating batch_index=0.0 specially is just to get scan_pessimistic_batches_for_oom()
            self.layers[i].layer_skip_rate = ScheduledFloat((cur_begin, initial_layerdrop_rate),
                                                            (cur_end, final_layerdrop_rate),
                                                            default=0.0)
            cur_begin = cur_end


    def forward(
        self,
        src: Tensor,
        feature_mask: Union[Tensor, float] = 1.0,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E)
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: (x, x_no_combine), both of shape (S, N, E)
        """
        pos_emb = self.encoder_pos(src)
        output = src


        rnd_seed = src.numel() + random.randint(0, 1000)

        output = output * feature_mask

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            output = output * feature_mask

        return output


class DownsampledZipformerEncoder(nn.Module):
    r"""
    DownsampledZipformerEncoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,
                 output_dim: int,
                 downsample: int):
        super(DownsampledZipformerEncoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = AttentionDownsample(input_dim, output_dim, downsample)
        self.encoder = encoder
        self.upsample = SimpleUpsample(output_dim, downsample)
        self.out_combiner = SimpleCombiner(input_dim,
                                           output_dim,
                                           min_weight=(0.0, 0.25))


    def forward(self,
                src: Tensor,
                feature_mask: Union[Tensor, float] = 1.0,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.  feature_mask is expected to be already downsampled by
               self.downsample_factor.
            mask: the mask for the src sequence (optional).  CAUTION: we need to downsample
                  this, if we are to support it.  Won't work correctly yet.
            src_key_padding_mask: the mask for the src keys per batch (optional).  Should
                  be downsampled already.

        Shape:
            src: (S, N, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: output of shape (S, N, F) where F is the number of output features
            (output_dim to constructor)
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if mask is not None:
            mask = mask[::ds,::ds]

        src = self.encoder(
            src, feature_mask=feature_mask, mask=mask, src_key_padding_mask=mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[:src_orig.shape[0]]

        return self.out_combiner(src_orig, src)


class DownsamplingZipformerEncoder(nn.Module):
    r"""
    DownsamplingZipformerEncoder is a zipformer encoder that downsamples its input
    by a specified factor before feeding it to the zipformer layers.
    """
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int,
                 output_dim: int,
                 downsample: int):
        super(DownsampledZipformerEncoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = AttentionDownsample(input_dim, output_dim, downsample)
        self.encoder = encoder


    def forward(self,
                src: Tensor,
                feature_mask: Union[Tensor, float] = 1.0,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer.  feature_mask is expected to be already downsampled by
               self.downsample_factor.
            mask: the mask for the src sequence (optional).  CAUTION: we need to downsample
                  this, if we are to support it.  Won't work correctly yet.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        Returns: output of shape (S, N, F) where F is the number of output features
            (output_dim to constructor)
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if mask is not None:
            mask = mask[::ds,::ds]
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask[::ds]

        src = self.encoder(
            src, feature_mask=feature_mask, mask=mask, src_key_padding_mask=mask,
        )
        return src


class AttentionDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: int):
        """
        Require out_channels > in_channels.
        """
        super(AttentionDownsample, self).__init__()
        self.query = nn.Parameter(torch.randn(in_channels) * (in_channels ** -0.5))

        # fill in the extra dimensions with a projection of the input
        if out_channels > in_channels:
            self.extra_proj = nn.Linear(in_channels * downsample,
                                        out_channels - in_channels,
                                        bias=False)
        else:
            self.extra_proj = None
        self.downsample = downsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, out_channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        if seq_len != d_seq_len * ds:
            # right-pad src, repeating the last element.
            pad = d_seq_len * ds - seq_len
            src_extra = src[src.shape[0]-1:].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
            assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        scores = (src * self.query).sum(dim=-1, keepdim=True)

        scores =  penalize_abs_values_gt(scores,
                                         limit=10.0,
                                         penalty=1.0e-04)

        weights = scores.softmax(dim=1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)
        src = src.permute(0, 2, 1, 3).reshape(d_seq_len, batch_size, ds * in_channels)

        if self.extra_proj is not None:
            ans2 = self.extra_proj(src)
            ans = torch.cat((ans, ans2), dim=2)
        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """
    def __init__(self,
                 num_channels: int,
                 upsample: int):
        super(SimpleUpsample, self).__init__()
        self.bias = nn.Parameter(torch.randn(upsample, num_channels) * 0.01)

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.bias.shape[0]
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src + self.bias.unsqueeze(1)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src

class SimpleCombiner(torch.nn.Module):
    """
    A very simple way of combining 2 vectors of 2 different dims, via a
    learned weighted combination in the shared part of the dim.
    Args:
         dim1: the dimension of the first input, e.g. 256
         dim2: the dimension of the second input, e.g. 384.
    The output will have the same dimension as dim2.
    """
    def __init__(self,
                 dim1: int,
                 dim2: int,
                 min_weight: Tuple[float, float] = (0., 0.)):
        super(SimpleCombiner, self).__init__()
        assert dim2 >= dim1
        initial_weight1 = 0.1
        self.weight1 = nn.Parameter(torch.full((dim2,), initial_weight1))
        self.min_weight = min_weight

    def forward(self,
                src1: Tensor,
                src2: Tensor) -> Tensor:
        """
        src1: (*, dim1)
        src2: (*, dim2)

        Returns: a tensor of shape (*, dim2)
        """
        assert src1.shape[:-1] == src2.shape[:-1]
        dim1 = src1.shape[-1]
        dim2 = src2.shape[-1]


        weight1 = self.weight1
        if self.training:
            weight1 = limit_param_value(weight1,
                                        min=self.min_weight[0],
                                        max=1.0-self.min_weight[1])

        src1_dim = src1.shape[-1]
        src2_dim = src2.shape[-1]
        if src1_dim != src2_dim:
            if src1_dim < src2_dim:
                zeros_shape = list(src1.shape[:-1]) + [src2_dim - src1_dim]
                src1 = torch.cat((src1, torch.zeros(*zeros_shape,
                                                    device=src1.device,
                                                    dtype=src1.dtype)),
                                 dim=-1)
            else:
                src1 = src1[:src2_dim]

        src1 = src1 * weight1
        src2 = src2 * (1.0 - weight1)

        return src1 + src2




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
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """
    def __init__(
        self, embed_dim: int,
            dropout_rate: float,
            max_len: int = 1000,
            length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))



    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(0) >= x.size(0) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(
                    x.device
                ):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        T = x.size(0)
        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T-1), T,
                         device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = (self.embed_dim ** 0.5)
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = compression_length * x.sign() * ((x.abs() + compression_length).log() - math.log(compression_length))

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan() # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)


    def forward(self, x: torch.Tensor) -> Tensor:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (time, batch, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            self.pe.size(0) // 2
            - x.size(0)
            + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)



class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
            self,
            embed_dim: int,
            pos_dim: int,
            num_heads: int,
            query_head_dim: int,
            pos_head_dim: int,
            dropout: float = 0.0,
            pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                          (4000.0, 0.0))
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(embed_dim, in_proj_dim, bias=True,
                                    initial_scale=query_head_dim**-0.25)

        # .. TODO: tune this limit? whitening_limit.
        self.whiten_keys = Whiten(num_groups=num_heads,
                                  whitening_limit=2.0,
                                  prob=(0.025, 0.25),
                                  grad_scale=0.025)


        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(pos_dim,
                                       num_heads * pos_head_dim,
                                       bias=False,
                                       initial_scale=0.05)


        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()


    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 2, pos_dim)
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[...,0:query_dim]
        k = x[...,query_dim:2*query_dim]
        # p is the position-encoding query
        p = x[...,2*query_dim:]
        assert p.shape[-1] == num_heads * pos_head_dim


        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(k)  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.


        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        if not self.training or random.random() >= float(self.pos_emb_skip_rate):
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            pos_scores = pos_scores.as_strided((num_heads, batch_size, seq_len, seq_len),
                                               (pos_scores.stride(0),
                                                pos_scores.stride(1),
                                                pos_scores.stride(2)-pos_scores.stride(3),
                                                pos_scores.stride(3)),
                                               storage_offset=pos_scores.stride(3) * (seq_len - 1))

            attn_scores = attn_scores + pos_scores

        if self.training and random.random() < 0.1:
            # This is a harder way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 50.0.  this should be outside the normal range
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
                                                 penalty=1.0e-04)

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            attn_scores.masked_fill_(attn_mask, float("-inf"))

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, seq_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                float("-inf"),
            )

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
                logging.info(f"attn_weights_entropy = {attn_weights_entropy}")


class SelfAttention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim,
                                 num_heads * value_head_dim,
                                 bias=True)

        self.out_proj = ScaledLinear(num_heads * value_head_dim,
                                     embed_dim, bias=True,
                                     initial_scale=0.05)

        # intended to prevent an observed failure mode where the output of this module is
        # dominated by its mean.
        self.out_balancer = ActivationBalancer(embed_dim,
                                               channel_dim=-1,
                                               min_positive=0.33,
                                               max_positive=0.66,
                                               min_abs=0.005, max_abs=1.0,
                                               min_prob=0.05)

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
          with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
          attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)     #  (seq_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(
            seq_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.out_balancer(x)

        return x


class AttentionSqueeze(nn.Module):
    """
    A modified version of Squeeze-and-Excite, where the nonliearity happens in the full dim and
    we just project to a small bottleneck dimension.
    """
    def __init__(self,
                 embed_dim: int,
                 bottleneck_dim: int = 16):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        self.in_proj = nn.Linear(embed_dim, embed_dim,
                                 bias=False)

        self.to_bottleneck_proj =  nn.Linear(embed_dim,
                                             bottleneck_dim,
                                             bias=False)


        # the main reason for this balancer is to keep the bottleneck activations in a "reasonable"
        # dynamic range, to avoid parameter-size 'drift' where to_bottleneck_proj gets large and from_bottleneck_proj
        # gets small or vice versa.
        # Caution: this cannot work correctly with an extremeley small batch size, e.g. if
        # we were training with a single very long audio sequence, or just 2 or 3 sequences
        # at a time.  We make max_factor small to reduce the harm this could cause
        # (although when the grads get back past the averaging operation they would
        # be quite small and would probably not hurt the rest of the model much.)
        self.bottleneck_balancer = ActivationBalancer(
            bottleneck_dim, channel_dim=-1,
            min_positive=0.05, max_positive=0.95,
            min_abs=0.05,
            max_abs=10.0,
            max_factor=0.02,
            min_prob=0.1,
        )
        self.bottleneck_activation = TanSwish()   # in bottleneck
        self.activation = Identity() # for diagnostics

        # the next two balancers are only to stop parameter-magnitude 'drift': we have
        # too many degrees of freedom for the scales of the various activations.
        # Make them run with very low probability, since only a small application of
        # these balancers should be enough to stop such "drift"; and, for speed,
        # put no limitation on the signs (so: min_positive=0, max_positive=1).
        self.scale_balancer = ActivationBalancer(
            embed_dim, channel_dim=-1,
            min_positive=0.2, max_positive=0.8,
            min_abs=0.2,  max_abs=1.0,
            min_prob=0.05,
        )
        self.activation_balancer = ActivationBalancer(
            embed_dim, channel_dim=-1,
            min_positive=0.2, max_positive=0.8,
            min_abs=0.2,  max_abs=1.0,
            min_prob=0.05,
        )

        self.from_bottleneck_proj =  ScaledLinear(bottleneck_dim, embed_dim)

        self.out_proj = ScaledLinear(embed_dim, embed_dim,
                                     bias=False, initial_scale=0.05)

        self.out_whiten = Whiten(num_groups=1,
                                 whitening_limit=10.0,
                                 prob=(0.01, 0.1),
                                 grad_scale=0.01)

    def forward(self,
                x: Tensor,
                attn_weights: Tensor):
        """
        Args:
           x: a Tensor of shape (seq_len, batch_size, num_channels)
attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
        Returns:
           a Tensor with the same shape as x
        """
        num_heads = attn_weights.shape[0]
        bottleneck = self.to_bottleneck_proj(x)  # (seq_len, batch_size, bottleneck_dim)
        (seq_len, batch_size, bottleneck_dim) = bottleneck.shape
        head_dim = bottleneck_dim // num_heads
        bottleneck = bottleneck.reshape(seq_len, batch_size, num_heads, head_dim).permute(
            2, 1, 0, 3)  # (num_heads, batch_size, seq_len, head_dim)

        # (num_heads, batch_size, seq_len, seq_len) x (num_heads, batch_size, seq_len, head_dim)
        #  -> (num_heads, batch_size, seq_len, head_dim)
        bottleneck = torch.matmul(attn_weights, bottleneck)
        bottleneck = self.bottleneck_balancer(bottleneck)
        bottleneck = self.bottleneck_activation(bottleneck)
        bottleneck = bottleneck.permute(2, 1, 0, 3) # (seq_len, batch_size, num_heads, head_dim)
        bottleneck = bottleneck.reshape(seq_len, batch_size, bottleneck_dim)
        scales = self.from_bottleneck_proj(bottleneck)

        x = self.in_proj(x)
        x = self.activation_balancer(x)
        scales = self.scale_balancer(scales)
        x = x * scales
        x = self.activation(x)  # Identity only.  For diagnostics.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer model.
    """
    def __init__(self,
                 embed_dim: int,
                 feedforward_dim: int,
                 dropout: float):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)
        self.hidden_balancer = ActivationBalancer(feedforward_dim,
                                                  channel_dim=-1, max_abs=10.0,
                                                  min_prob=0.25)
        self.activation = DoubleSwish()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = ScaledLinear(feedforward_dim, embed_dim,
                                     initial_scale=0.01)
        self.out_whiten =  Whiten(num_groups=1,
                                  whitening_limit=10.0,
                                  prob=(0.025, 0.25),
                                  grad_scale=0.01)

    def forward(self,
                x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttentionModule(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self, channels: int,
    ) -> None:
        super().__init__()

        self.in_proj = nn.Linear(channels, 2 * channels, bias=True)

        # balancer that goes after the glu mechanism.
        self.balancer = ActivationBalancer(
            channels, channel_dim=-1,
            min_positive=0.2, max_positive=0.8,
            min_abs=0.2, max_abs=10.0,
            min_prob=0.05,
        )
        # give it a high limit, because it is quite high-dimensional and is
        # a projection of a lower-dimensional embedding.
        self.whiten = Whiten(num_groups=2,
                             whitening_limit=20.0,
                             prob=(0.025, 0.25),
                             grad_scale=0.01)


        self.activation = Identity()  # for diagnostics.
        self.out_proj = ScaledLinear(channels, channels,
                                     bias=True,
                                     initial_scale=0.05)


        # put quite strict limits on the min_positive and max_positive at the output,
        # because we noticed that poorly-trained instances of NonlinAttentionModule seem
        # to have a larger mean-offset at the output for some reason.
        self.out_balancer = ActivationBalancer(
            channels, channel_dim=-1,
            min_positive=0.45, max_positive=0.55,
            min_abs=0.005, max_abs=1.0,
            min_prob=0.05,
        )


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
        x = self.in_proj(x)
        x = self.whiten(x)
        v, s = x.chunk(2, dim=-1)

        if self.training and random.random() < 0.02:
            # prevent the inputs to the sigmoid from getting very large (this is
            # hopefully quite a rare phenomenon, so we are giving this path a
            # very small probability to save time).
            s = penalize_abs_values_gt(s, limit=20.0, penalty=1.0e-04)

        # GLU mechanism
        x = s.sigmoid() * v
        x = self.balancer(x)

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        x = self.activation(x)  # diagnostics only, it's the identity.
        x = self.out_proj(x)
        x = self.out_balancer(x)
        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self, channels: int, kernel_size: int, bias: bool = True
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # after pointwise_conv1 we put x through a gated linear unit (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
        self.deriv_balancer1 = ActivationBalancer(
            2 * channels,
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
        )

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )

        self.deriv_balancer2 = ActivationBalancer(
            channels, channel_dim=1,
            min_positive=0.05, max_positive=1.0,
            max_abs=20.0,
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.05,
        )

    def forward(self,
                x: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains bool in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)

        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        if src_key_padding_mask is not None:
            x.masked_fill_(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = (T-3)//2 - 2 == (T-7)//2

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, (T-3)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
        """
        assert in_channels >= 7
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),  # (time, freq)
            ),
            ActivationBalancer(layer1_channels,
                               channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            ActivationBalancer(layer2_channels,
                               channel_dim=1),
            DoubleSwish(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=(1, 2), # (time, freq)
            ),
            ActivationBalancer(layer3_channels,
                               channel_dim=1),
            DoubleSwish(),
        )
        out_height = (((in_channels - 1) // 2) - 1) // 2
        self.out = ScaledLinear(out_height * layer3_channels, out_channels)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-3)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.dropout(x)
        return x

class AttentionCombine(nn.Module):
    """
    This module combines a list of Tensors, all with the same shape, to
    produce a single output of that same shape which, in training time,
    is a random combination of all the inputs; but which in test time
    will be just the last input.

    All but the last input will have a linear transform before we
    randomly combine them; these linear transforms will be initialized
    to the identity transform.

    The idea is that the list of Tensors will be a list of outputs of multiple
    zipformer layers.  This has a similar effect as iterated loss. (See:
    DEJA-VU: DOUBLE FEATURE PRESENTATION AND ITERATED LOSS IN DEEP TRANSFORMER
    NETWORKS).
    """

    def __init__(
        self,
        num_channels: int,
        num_inputs: int,
        random_prob: float = 0.25,
        single_prob: float = 0.333,
    ) -> None:
        """
        Args:
          num_channels:
            the number of channels
          num_inputs:
            The number of tensor inputs, which equals the number of layers'
            outputs that are fed into this module.  E.g. in an 18-layer neural
            net if we output layers 16, 12, 18, num_inputs would be 3.
          random_prob:
            the probability with which we apply a nontrivial mask, in training
            mode.
         single_prob:
            the probability with which we mask to allow just a single
            module's output (in training)
        """
        super().__init__()

        self.random_prob = random_prob
        self.single_prob = single_prob
        self.weight  = torch.nn.Parameter(torch.zeros(num_channels,
                                                     num_inputs))
        self.bias = torch.nn.Parameter(torch.zeros(num_inputs))

        assert 0 <= random_prob <= 1, random_prob
        assert 0 <= single_prob <= 1, single_prob



    def forward(self, inputs: List[Tensor]) -> Tensor:
        """Forward function.
        Args:
          inputs:
            A list of Tensor, e.g. from various layers of a transformer.
            All must be the same shape, of (*, num_channels)
        Returns:
          A Tensor of shape (*, num_channels).  In test mode
          this is just the final input.
        """
        num_inputs = self.weight.shape[1]
        assert len(inputs) == num_inputs

        # Shape of weights: (*, num_inputs)
        num_channels = inputs[0].shape[-1]
        num_frames = inputs[0].numel() // num_channels

        ndim = inputs[0].ndim
        # stacked_inputs: (num_frames, num_channels, num_inputs)
        stacked_inputs = torch.stack(inputs, dim=ndim).reshape(
            (num_frames, num_channels, num_inputs)
        )

        scores = (stacked_inputs * self.weight).sum(dim=(1,)) + self.bias

        if random.random() < 0.002:
            logging.info(f"Average scores are {scores.softmax(dim=1).mean(dim=0)}")

        if self.training:
            # random masking..
            mask_start = torch.randint(low=1, high=int(num_inputs / self.random_prob),
                                       size=(num_frames,), device=scores.device).unsqueeze(1)
            # mask will have rows like: [ False, False, False, True, True, .. ]
            arange = torch.arange(num_inputs, device=scores.device).unsqueeze(0).expand(
                num_frames, num_inputs)
            mask = arange >= mask_start

            apply_single_prob = torch.logical_and(torch.rand(size=(num_frames, 1),
                                                             device=scores.device) < self.single_prob,
                                                  mask_start < num_inputs)
            single_prob_mask = torch.logical_and(apply_single_prob,
                                                 arange < mask_start - 1)

            mask = torch.logical_or(mask,
                                    single_prob_mask)

            scores = scores.masked_fill(mask, float('-inf'))

        if self.training and random.random() < 0.1:
            scores =  penalize_abs_values_gt(scores,
                                             limit=10.0,
                                             penalty=1.0e-04)

        weights = scores.softmax(dim=1)

        # (num_frames, num_channels, num_inputs) * (num_frames, num_inputs, 1) -> (num_frames, num_channels, 1),
        ans = torch.matmul(stacked_inputs, weights.unsqueeze(2))
        # ans: (*, num_channels)
        ans = ans.reshape(*tuple(inputs[0].shape[:-1]), num_channels)

        if __name__ == "__main__":
            # for testing only...
            print("Weights = ", weights.reshape(num_frames, num_inputs))
        return ans



def _test_random_combine():
    print("_test_random_combine()")
    num_inputs = 3
    num_channels = 50
    m = AttentionCombine(
        num_channels=num_channels,
        num_inputs=num_inputs,
        random_prob=0.5,
        single_prob=0.0)


    x = [torch.ones(3, 4, num_channels) for _ in range(num_inputs)]

    y = m(x)
    assert y.shape == x[0].shape
    assert torch.allclose(y, x[0])  # .. since actually all ones.


def _test_zipformer_main():
    feature_dim = 50
    batch_size = 5
    seq_len = 20
    feature_dim = 50
    # Just make sure the forward pass runs.

    c = Zipformer(
        num_features=feature_dim, encoder_dim=(64,96), encoder_unmasked_dim=(48,64), num_heads=(4,4)
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_random_combine()
    _test_zipformer_main()
