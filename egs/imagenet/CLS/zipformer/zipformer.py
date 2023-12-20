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
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from scaling import (
    Balancer,
    BiasNorm,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
    Dropout2,
)
from torch import Tensor, nn


class Zipformer2(nn.Module):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

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
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.  Must not be less than cnn_module_kernel (after factoring in
           rounding and downsampling); an error will be thrown if this is violated.
    """
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            downsampling_factor: Tuple[int] = (1, 2, 2, 2),
            encoder_dim: Union[int, Tuple[int]] = 384,
            num_encoder_layers: Union[int, Tuple[int]] = 4,
            query_head_dim: Union[int, Tuple[int]] = 32,
            value_head_dim: Union[int, Tuple[int]] = 12,
            num_heads: Union[int, Tuple[int]] = 8,
            feedforward_dim: Union[int, Tuple[int]] = 1536,
            cnn_module_kernel: Union[int, Tuple[int]] = 5,
            block_size: Union[int, Tuple[int]] = 4,
            shift_block: bool = False,
            dilate_block: bool = False,
            dropout: FloatLike = None,  # see code below for default
            warmup_batches: float = 4000.0,
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3),
                                     (20000.0, 0.1))

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

        self.downsampling_factor = downsampling_factor # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim) # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=encoder_dim[0])

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []
        cur_img_size = img_size // patch_size
        last_embed_dim = encoder_dim[0]
        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            cur_img_size = cur_img_size // downsampling_factor[i]
            assert cur_img_size % block_size[i] == 0
            # assert block_size[i] % 2 == 0, block_size[i]

            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                block_size=block_size[i],
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                resolution=(cur_img_size, cur_img_size),
                encoder_layer=encoder_layer,
                num_layers=num_encoder_layers[i],
                block_size=block_size[i],
                shift_block=shift_block if cur_img_size > block_size[i] else False,
                dilate_block=dilate_block if cur_img_size > block_size[i] else False,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    in_channel=last_embed_dim,
                    out_channel=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                )

            last_embed_dim = encoder_dim[i]
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # TODO: test removing it
        self.norm = BiasNorm(last_embed_dim)
        self.head = nn.Linear(last_embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x:
            The input image of shape (batch_size, in_chans, height, width).

        Returns:
            - embeddings: its shape is (batch_size, num_classes)
        """
        x = self.patch_embed(x)  # (batch, height, width, channel)

        for i, module in enumerate(self.encoders):
            x = module(x)

        x = self.norm(x)
        x = self.head(x.mean(dim=(1, 2)))

        return x


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x),
                          (20000.0, ratio * x),
                          default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))


class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
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
            feedforward_dim: int,
            block_size: int = 4,
            select_topk: int = 16,
            dropout: FloatLike = 0.1,
            cnn_module_kernel: int = 3,
            attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
            const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
            ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
            bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.02), default=0),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(embed_dim, skip_rate=bypass_skip_rate,
                                   straight_through_rate=0)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)

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

        self.self_attn_weights = MultiheadAttentionWeights(
            embed_dim, num_heads=num_heads,
            query_head_dim=query_head_dim,
            block_size=block_size, select_topk=select_topk,
            dropout=0.0,
        )

        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim,
                                        block_size=block_size, select_topk=select_topk)

        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim,
                                        block_size=block_size, select_topk=select_topk)

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
                                                hidden_channels=3 * embed_dim // 4,
                                                block_size=block_size, select_topk=select_topk)

        self.conv_module1 = ConvolutionModule(embed_dim,
                                              cnn_module_kernel)

        self.conv_module2 = ConvolutionModule(embed_dim,
                                              cnn_module_kernel)

        self.norm = BiasNorm(embed_dim)

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

        self.pos_dropout = Dropout2(p=0.1)

    def get_sample_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return None
        batch_size = x.shape[0]
        mask = (torch.rand(batch_size, 1, 1, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sample_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sample-level dropout to x.
        x shape: (batch_size, height, width, channel)
        """
        dropout_mask = self.get_sample_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

    def forward(
        self,
        src: Tensor,
        reordered_indexes: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder (required): shape (batch_size, height, width, channel)
            reordered_indexes (Optional): (1, height, width, 1)

        Returns:
           A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attention_skip_rate = 0.0
        else:
            attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # (num_heads, batch_size, num_block_tot, block_size ** 2, block_size ** 2 + select_topk)
        attn_weights = self.self_attn_weights(
            src, reordered_indexes=reordered_indexes
        )

        src = src + self.feed_forward1(src)

        self_attn_dropout_mask = self.get_sample_dropout_mask(src, attention_skip_rate)

        selected_attn_weights = attn_weights[0:1]
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif not self.training and random.random() < float(self.const_attention_rate):
            # Make attention weights constant.  The intention is to
            # encourage these modules to do something similar to an
            # averaging-over-time operation.
            # only need the mask, can just use the 1st one and expand later
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
            selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))

        na = self.balancer_na(self.nonlin_attention(
            src, reordered_indexes=reordered_indexes, attn_weights=selected_attn_weights))

        src = src + (na if self_attn_dropout_mask is None else na * self_attn_dropout_mask)

        self_attn = self.self_attn1(src, reordered_indexes=reordered_indexes, attn_weights=attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sample_dropout(self.conv_module1(src), conv_skip_rate)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff2_skip_rate = 0.0
        else:
            ff2_skip_rate = float(self.ff2_skip_rate) if self.training else 0.0
        src = src + self.sample_dropout(self.balancer_ff2(self.feed_forward2(src)),
                                        ff2_skip_rate)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(src, reordered_indexes=reordered_indexes, attn_weights=attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sample_dropout(self.conv_module2(src), conv_skip_rate)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff3_skip_rate = 0.0
        else:
            ff3_skip_rate = float(self.ff3_skip_rate) if self.training else 0.0
        src = src + self.sample_dropout(self.balancer_ff3(self.feed_forward3(src)),
                                        ff3_skip_rate)

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src


class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
       pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """
    def __init__(
            self,
            resolution: Tuple[int, int],
            encoder_layer: nn.Module,
            num_layers: int,
            dropout: float,
            warmup_begin: float,
            warmup_end: float,
            block_size: int = 4,
            shift_block: bool = True,
            dilate_block: bool = False,
            initial_layerdrop_rate: float = 0.5,
            final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()

        self.name = None

        self.resolution = resolution
        self.block_size = block_size

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.embed_dim = encoder_layer.embed_dim
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end

        delta = (1. / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat((cur_begin, initial_layerdrop_rate),
                                                             (cur_end, final_layerdrop_rate),
                                                             default=0.0)
            cur_begin = cur_end

        if shift_block:
            shifted_indexes = self.generate_shifted_indexes()
        else:
            shifted_indexes = None
        self.register_buffer("shifted_indexes", shifted_indexes)

        if dilate_block:
            dilated_indexes = self.generate_dilated_indexes()
        else:
            dilated_indexes = None
        self.register_buffer("dilated_indexes", dilated_indexes)

    def generate_shifted_indexes(self):
        # (height, width)
        resolution = self.resolution
        indexes = torch.arange(resolution[0] * resolution[1]).reshape(*resolution)
        shift = self.block_size // 2
        # (1, height * width, 1)
        shifted_indexes = indexes.roll(shifts=(shift, shift), dims=(0, 1))
        shifted_indexes = shifted_indexes.flatten().unsqueeze(0).unsqueeze(-1)
        return shifted_indexes

    def generate_dilated_indexes(self):
        # (height, width)
        resolution = self.resolution
        indexes = torch.arange(resolution[0] * resolution[1]).reshape(*resolution)
        indexes = torch.stack([
            indexes[::2, ::2], indexes[::2, 1::2], indexes[1::2, ::2], indexes[1::2, 1::2]
        ], dim=0)
        indexes = indexes.view(2, 2, resolution[0] // 2, resolution[1] // 2).permute(0, 2, 1, 3)
        # now indexes: (2, resolution[0] // 1, 2, resolution[1])
        # (1, height * width, 1)
        dilated_indexes = indexes.flatten().unsqueeze(0).unsqueeze(-1)
        return dilated_indexes

    def forward(
        self,
        src: Tensor,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
        Returns: a Tensor with the same shape as src.
        """
        batch, height, width, channel = src.size()

        assert (height, width) == self.resolution

        output = src

        reordered_indexes_list = [None]
        # reordered_indexes = self.shifted_indexes if self.shifted_indexes is not None else self.dilated_indexes
        if self.shifted_indexes is not None:
            reordered_indexes_list.append(self.shifted_indexes)
        if self.dilated_indexes is not None:
            reordered_indexes_list.append(self.dilated_indexes)

        num_types = len(reordered_indexes_list)
        # assert self.num_layers >= num_types
        for i, mod in enumerate(self.layers):
            # shift blocks between successive layers
            output = mod(
                output,
                reordered_indexes=reordered_indexes_list[i % num_types],
            )

        return output


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

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale,
                                    min=float(self.scale_min),
                                    max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1, 1, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1, 1, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self,
                src_orig: Tensor,
                src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[0])
        return src_orig + (src - src_orig) * bypass_scale


class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """
    def __init__(self,
                 encoder: nn.Module,
                 in_channel: int,
                 out_channel: int,
                 downsample: int,
                 dropout: FloatLike):
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = Downsample(in_channel, out_channel, downsample, dropout)
        self.num_layers = encoder.num_layers
        self.encoder = encoder

    def forward(
        self, src: Tensor,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required): shape (batch_size, height, width, embedding_dim).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (batch_size, 1, 1, embedding_dim)

        Returns: a Tensor with the same shape as src.
        """
        src = self.downsample(src)
        src = self.encoder(src)

        return src


class Downsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 downsample: int,
                 dropout: FloatLike):
        super(Downsample, self).__init__()

        # self.bias = nn.Parameter(torch.zeros(downsample ** 2))

        self.name = None  # will be set from training code

        self.downsample = downsample

        self.reduction = nn.Linear(downsample ** 2 * in_channel, out_channel, bias=False)

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (batch, height, width, in_channel)
        Returns a tensor of shape
           (batch, height // ds, width // ds, out_channel)
        """
        batch, height, width, channel = src.shape
        ds = self.downsample
        assert width % ds == 0 and height % ds == 0, (height, width)
        d_height = height // ds
        d_width = width // ds

        src = src.reshape(batch, d_height, ds, d_width, ds, channel).permute(0, 1, 3, 2, 4, 5)
        # src = src.reshape(batch, d_height, d_width, ds * ds, channel)

        # weights = self.bias.softmax(dim=0)
        # # weights: (ds * ds , 1)
        # weights = weights.unsqueeze(-1)

        # # (batch, d_height, d_width, channel)
        # ans = (src * weights).sum(dim=3)

        src = src.reshape(batch, d_height, d_width, ds * ds * channel)
        src = self.reduction(src)

        return src


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """
    def __init__(self,
                 num_channels: int,
                 upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self,
                src: Tensor) -> Tensor:
        """
        x: (batch, height, width, channel)
        Returns a tensor of shape
           (batch, height * upsample, width * upsample, channel)
        """
        upsample = self.upsample
        batch, height, width, channel = src.shape
        src = src.unsqueeze(2).unsqueeze(4).expand(
            batch, height, upsample, width, upsample, channel)
        src = src.reshape(batch, height * upsample, width * upsample, channel)
        return src


class MultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
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
            num_heads: int,
            query_head_dim: int,
            block_size: int,
            select_topk: int,
            dropout: float = 0.0,
            pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                          (4000.0, 0.0))
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim

        self.block_size = block_size
        self.select_topk = select_topk

        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim) * num_heads

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

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.rand((2 * block_size - 1) * (2 * block_size - 1), num_heads) * .02)  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(block_size)
        coords_w = torch.arange(block_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += block_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += block_size - 1
        relative_coords[:, :, 0] *= 2 * block_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(
        self,
        x: Tensor,
        reordered_indexes: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (batch_size, height, width, channel)
            reordered_indexes (Optional): (1, height, width, 1)
        Returns:
           a tensor of attention weights, of shape
           (num_heads, batch_size, num_block_tot, block_size ** 2, block_size ** 2)
        """
        batch_size, height, width, channel = x.shape
        block_size = self.block_size
        assert height % block_size == 0 and width % block_size == 0, (height, width, block_size)

        num_tokens = height * width  # total number of tokens
        num_block_h = height // block_size
        num_block_w = width // block_size
        num_block_tot = num_block_h * num_block_w

        if reordered_indexes is not None:
            x = x.reshape(batch_size, num_tokens, channel)
            assert reordered_indexes.shape == (1, num_tokens, 1)
            reordered_indexes = reordered_indexes.expand(batch_size, num_tokens, channel)
            reordered_x = torch.gather(x, dim=1, index=reordered_indexes)
            x = reordered_x.view(batch_size, height, width, channel)

        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        num_heads = self.num_heads
        query_dim = query_head_dim * num_heads

        # self-attention
        query = x[...,0:query_dim]
        key = x[...,query_dim:2*query_dim]

        query = self.copy_query(query)  # for diagnostics only, does nothing.
        key = self.whiten_keys(self.balance_keys(key))  # does nothing in the forward pass.

        query = query.reshape(
            batch_size, num_block_h, block_size, num_block_w, block_size, num_heads, query_head_dim)
        query = query.permute(5, 0, 1, 3, 2, 4, 6)
        # now query: (head, batch, num_block_h, num_block_w, block_size, block_size, query_head_dim)
        query = query.reshape(num_heads, batch_size, num_block_tot, block_size ** 2, query_head_dim)

        key = key.reshape(
            batch_size, num_block_h, block_size, num_block_w, block_size, num_heads, query_head_dim)
        key = key.permute(5, 0, 1, 3, 6, 2, 4)
        # now key: (head, batch, num_block_h, num_block_w, query_head_dim, block_size, block_size)
        key = key.reshape(num_heads, batch_size, num_block_tot, query_head_dim, block_size ** 2)

        # (num_heads, batch_size, num_block_tot, block_size**2, block_size**2)
        attn_scores = torch.matmul(query, key)

        if random.random() >= float(self.pos_emb_skip_rate):
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(block_size ** 2, block_size ** 2, num_heads)  # (block_size**2,block_size**2,num_heads)
            # (num_heads, block_size**2, block_size**2)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            # (num_heads, 1, 1, block_size**2, block_size**2)
            relative_position_bias = relative_position_bias.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores + relative_position_bias

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < 0.1:
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
                                                 penalty=1.0e-04,
                                                 name=self.name)

        assert attn_scores.shape == (
            num_heads, batch_size, num_block_tot, block_size ** 2, block_size ** 2)

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001 and not self.training:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights

    def _print_attn_entropy(self, attn_weights: Tensor):
        # (num_heads, batch_size, num_block_tot, block_size**2, block_size**2+select_topk)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(
                    dim=-1).mean(dim=(1, 2, 3))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")


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
            block_size: int,
            select_topk: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.value_head_dim = value_head_dim
        self.block_size = block_size
        self.select_topk = select_topk

        self.in_proj = nn.Linear(embed_dim,
                                 num_heads * value_head_dim,
                                 bias=True)

        self.out_proj = ScaledLinear(num_heads * value_head_dim,
                                     embed_dim, bias=True,
                                     initial_scale=0.05)

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5, ratio=3.0),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        reordered_indexes: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: input tensor, of shape (batch_size, height, width, channel)
            reordered_indexes (Optional): (1, height, width, 1)
            attn_weights: a tensor of shape
              (num_heads, batch_size, num_block_tot, block_size ** 2, block_size ** 2)
              Expect attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (batch_size, height, width, channel) = x.shape

        num_heads = self.num_heads
        value_head_dim = self.value_head_dim
        value_dim = num_heads * value_head_dim

        block_size = self.block_size
        assert height % block_size == 0 and width % block_size == 0, (height, width, block_size)

        num_tokens = height * width  # total number of tokens
        num_block_h = height // block_size
        num_block_w = width // block_size
        num_block_tot = num_block_h * num_block_w

        assert attn_weights.shape == (
            num_heads, batch_size, num_block_tot, block_size ** 2, block_size ** 2)

        if reordered_indexes is not None:
            x = x.reshape(batch_size, num_tokens, -1)
            assert reordered_indexes.shape == (1, height * width, 1)
            reordered_indexes = reordered_indexes.expand(batch_size, num_tokens, channel)
            reordered_x = torch.gather(x, dim=1, index=reordered_indexes)
            x = reordered_x.view(batch_size, height, width, channel)

        x = self.in_proj(x)  # (batch_size, height, width, num_heads * value_head_dim)

        x = x.reshape(
            batch_size, num_block_h, block_size, num_block_w, block_size, num_heads, value_head_dim)
        x = x.permute(5, 0, 1, 3, 2, 4, 6)
        # now x: (head, batch, num_block_h, num_block_w, block_size, block_size, value_head_dim)
        x = x.reshape(num_heads, batch_size, num_block_tot, block_size ** 2, value_head_dim)

        # (head, batch, num_block_tot, block_size**2, value_head_dim)
        x = torch.matmul(attn_weights, x)

        x = x.reshape(
            num_heads, batch_size, num_block_h, num_block_w, block_size, block_size, value_head_dim)
        x = x.permute(1, 2, 4, 3, 5, 0, 6)
        # now: (batch, num_block_h, block_size, num_block_w, block_size, head, value_head_dim)
        x = x.reshape(batch_size, height, width, value_dim)

        # returned value is of shape (batch, height, width, channel), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        if reordered_indexes is not None:
            # recover to original order
            x = x.reshape(batch_size, height * width, channel)
            new_x = torch.zeros_like(x)
            new_x.scatter_(dim=1, index=reordered_indexes, src=x)
            x = new_x.reshape(batch_size, height, width, channel)

        return x


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model.
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

        self.out_whiten = Whiten(num_groups=1,
                                 whitening_limit=_whitening_schedule(7.5),
                                 prob=(0.025, 0.25),
                                 grad_scale=0.01)

    def forward(self, x: Tensor):
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
            block_size: int,
            select_topk: int,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.block_size = block_size
        self.select_topk = select_topk

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

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        reordered_indexes: Optional[Tensor] = None,
    ) -> Tensor:
        """.
        Args:
            x: a Tensor of shape (seq_len, batch_size, num_channels)
            reordered_indexes (Optional): (1, height, width, 1)
            attn_weights: a tensor of shape
              (num_heads, batch_size, num_block_tot, block_size ** 2, block_size ** 2 + select_topk)
              Expect attn_weights.sum(dim=-1) == 1.
        Returns:
           a Tensor with the same shape as x
        """
        (batch_size, height, width, channel) = x.shape
        block_size = self.block_size
        num_tokens = height * width  # total number of tokens
        num_block_h = height // block_size
        num_block_w = width // block_size
        num_block_tot = num_block_h * num_block_w
        assert height % block_size == 0 and width % block_size == 0, (height, width, block_size)

        if reordered_indexes is not None:
            x = x.reshape(batch_size, num_tokens, -1)
            assert reordered_indexes.shape == (1, height * width, 1)
            reordered_indexes = reordered_indexes.expand(batch_size, num_tokens, channel)
            reordered_x = torch.gather(x, dim=1, index=reordered_indexes)
            x = reordered_x.view(batch_size, height, width, -1)

        x = self.in_proj(x)

        s, x, y = x.chunk(3, dim=-1)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.
        # now x: (batch, height, width, channel)

        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (
            num_heads, batch_size, num_block_tot, block_size ** 2, block_size ** 2)

        hidden_dim = self.hidden_channels
        head_dim = hidden_dim // num_heads

        x = x.reshape(
            batch_size, num_block_h, block_size, num_block_w, block_size, num_heads, head_dim)
        x = x.permute(5, 0, 1, 3, 2, 4, 6)
        # now x: (head, batch, num_block_h, num_block_w, block_size, block_size, head_dim)
        x = x.reshape(num_heads, batch_size, num_block_tot, block_size ** 2, head_dim)

        # (head, batch, num_block_tot, block_size**2, head_dim)
        x = torch.matmul(attn_weights, x)

        x = x.reshape(
            num_heads, batch_size, num_block_h, num_block_w, block_size, block_size, head_dim)
        x = x.permute(1, 2, 4, 3, 5, 0, 6)
        # now: (batch, num_block_h, block_size, num_block_w, block_size, head, head_dim)
        x = x.reshape(batch_size, height, width, hidden_dim)

        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)

        if reordered_indexes is not None:
            # recover to original order
            x = x.reshape(batch_size, height * width, channel)
            new_x = torch.zeros_like(x)
            new_x.scatter_(dim=1, index=reordered_indexes, src=x)
            x = new_x.reshape(batch_size, height, width, channel)

        return x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """
    def __init__(
        self, channels: int, kernel_size: int = 3,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels

        self.in_proj = nn.Linear(
            channels, 2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        # after in_proj we put x through a gated linear unit (nn.functional.glu).
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
        self.balancer1 = Balancer(
            bottleneck_dim, channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity()  # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity()  # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = nn.Conv2d(
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            groups=bottleneck_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2)

        self.balancer2 = Balancer(
            bottleneck_dim, channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(num_groups=1,
                             whitening_limit=_whitening_schedule(7.5),
                             prob=(0.025, 0.25),
                             grad_scale=0.01)

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim, channels, activation='SwooshR',
            dropout_p=0.0, initial_scale=0.05,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (batch, height, width, channel).

        Returns:
            Tensor: Output tensor (batch, height, width, channel).

        """

        x = self.in_proj(x)  # (batch, height, width, 2*channels)

        x, s = x.chunk(2, dim=-1)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # now x: (batch, height, width, channel)

        # exchange the spatial dimension and the feature dimension
        x = x.permute(0, 3, 1, 2)  # (batch, channel, height, width)

        x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channel)

        x = self.whiten(x)  # (batch, height, width, channel)
        x = self.out_proj(x)  # (batch, height, width, channel)

        return x


class ScalarMultiply(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


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
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = BiasNorm(embed_dim, channel_dim=-1)
        self.out_balancer = Balancer(
            embed_dim, channel_dim=-1, min_positive=0.45, max_positive=0.55
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.out_balancer(x)
        x = self.norm(x)
        return x


def _test_zipformer_main(shift_block: bool = True, dilate_block: bool = False):
    # Just make sure the forward pass runs.

    c = Zipformer2(
        patch_size=4,
        encoder_dim=(64,128,256,512),
        downsampling_factor=(1,2,2,2),
        num_encoder_layers=(2,5,4,2),
        feedforward_dim=(192,384,768,1536),
        num_heads=(2,4,8,16),
        cnn_module_kernel=(7,7,5,3),
        block_size=(7,7,7,7),
        shift_block=shift_block,
        dilate_block=dilate_block,
    )
    num_param = sum([p.numel() for p in c.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    batch_size = 5
    # Just make sure the forward pass runs.
    f = c(torch.randn(batch_size, 3, 224, 224))
    f.sum().backward()
    c.eval()
    f = c(torch.randn(batch_size, 3, 224, 224))
    f  # to remove flake8 warnings


def _test_shifted_indexes():
    # shifted indxes
    resolution = (8, 8)
    indexes = torch.arange(resolution[0] * resolution[1]).reshape(*resolution)
    shift = 2
    # (1, height * width, 1)
    shifted_indexes = indexes.roll(shifts=(shift, shift), dims=(0, 1))
    shifted_indexes = shifted_indexes.flatten().unsqueeze(0).unsqueeze(-1)

    batch_size = 2
    num_tokens = resolution[0] * resolution[1]
    channel = 32
    x = torch.randn(batch_size, num_tokens, channel)

    reordered_indexes = shifted_indexes
    reordered_indexes = reordered_indexes.expand(batch_size, num_tokens, channel)
    reordered_x = torch.gather(x, dim=1, index=reordered_indexes)

    new_x = torch.zeros_like(x)
    new_x.scatter_(dim=1, index=reordered_indexes, src=reordered_x)

    assert torch.allclose(new_x, x)


def _test_ditaled_indexes():
    # dilated indxes
    resolution = (8, 8)
    indexes = torch.arange(resolution[0] * resolution[1]).reshape(*resolution)
    indexes = torch.stack([
        indexes[::2, ::2], indexes[::2, 1::2], indexes[1::2, ::2], indexes[1::2, 1::2]
    ], dim=0)
    indexes = indexes.view(2, 2, resolution[0] // 2, resolution[1] // 2).permute(0, 2, 1, 3)
    # now indexes: (2, resolution[0] // 1, 2, resolution[1])
    # (1, height * width, 1)
    dilated_indexes = indexes.flatten().unsqueeze(0).unsqueeze(-1)

    batch_size = 2
    num_tokens = resolution[0] * resolution[1]
    channel = 32
    x = torch.randn(batch_size, num_tokens, channel)

    reordered_indexes = dilated_indexes
    reordered_indexes = reordered_indexes.expand(batch_size, num_tokens, channel)
    reordered_x = torch.gather(x, dim=1, index=reordered_indexes)

    new_x = torch.zeros_like(x)
    new_x.scatter_(dim=1, index=reordered_indexes, src=reordered_x)

    assert torch.allclose(new_x, x)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # _test_shifted_indexes()
    # _test_ditaled_indexes()
    _test_zipformer_main(False, False)
    # _test_zipformer_main(True, False)
    # _test_zipformer_main(False, True)
    # _test_zipformer_main(True, True)
