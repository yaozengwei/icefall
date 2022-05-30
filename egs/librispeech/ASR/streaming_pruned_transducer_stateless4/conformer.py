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

import logging
import copy
import math
import warnings
from typing import List, Optional, Tuple, Set

import torch
from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
)
from torch import Tensor, nn

from icefall.utils import make_pad_mask


# Copied and modified from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/mask.py
def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def chunk_mask_with_left_context(
    chunk_size: int,
    left_context_size: int,
    cached_left_context_sizes: Tensor,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Obtain mask for streaming decoding.
    Args:
      chunk_size:
        Length of chunk frames.
      left_context_size:
        Target length of left context frames.
      cached_left_context_sizes:
        Real lengths of cached left context frames for each sample in the batch.
        Its size is (batch_size,).
      device:
        "cpu" or "cuda" or torch.Tensor.device.

    Returns:
      A tensor of shape
      (batch_size, chunk_length, target_left_context_length + chunk_length).
    """
    batch_size = len(cached_left_context_sizes)
    mask = torch.zeros(
        chunk_size,
        left_context_size + chunk_size,
        device=device,
        dtype=torch.bool,
    )
    for i in range(batch_size):
        cached_left = cached_left_context_sizes[i].item()
        assert cached_left <= left_context_size
        mask[i, :, -(cached_left + chunk_size) :] = True
    return mask


class Conformer(EncoderInterface):
    """
    Args:
        num_features (int): Number of input features
        subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
        d_model (int): attention dimension, also the output dimension
        nhead (int): number of head
        dim_feedforward (int): feedforward dimention
        num_encoder_layers (int): number of encoder layers
        dropout (float): dropout rate
        layer_dropout (float): layer-dropout rate.
        cnn_module_kernel (int): Kernel size of convolution module
        vgg_frontend (bool): whether to use vgg frontend.
        dynamic_chunk_training (bool): whether to use dynamic chunk training, if
            you want to train a streaming model, this is expected to be True.
            When setting True, it will use a masking strategy to make the attention
            see only limited left and right context.
        short_chunk_threshold (float): a threshold to determinize the chunk size
            to be used in masking training, if the randomly generated chunk size
            is greater than ``max_len * short_chunk_threshold`` (max_len is the
            max sequence length of current batch) then it will use
            full context in training (i.e. with chunk size equals to max_len).
            This will be used only when dynamic_chunk_training is True.
        short_chunk_size (int): see docs above, if the randomly generated chunk
            size equals to or less than ``max_len * short_chunk_threshold``, the
            chunk size will be sampled uniformly from 1 to short_chunk_size.
            This also will be used only when dynamic_chunk_training is True.
        num_left_chunks (int): the left context (in chunks) attention can see, the
            chunk size is decided by short_chunk_threshold and short_chunk_size.
            A minus value means seeing full left context.
            This also will be used only when dynamic_chunk_training is True.
        causal (bool): Whether to use causal convolution in conformer encoder
            layer. This MUST be True when using dynamic_chunk_training.
    """

    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 31,
        short_chunk_threshold: float = 0.75,
        short_chunk_size: int = 25,
        num_left_chunks: int = -1,
    ) -> None:
        super(Conformer, self).__init__()

        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.num_encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.dynamic_chunk_training = dynamic_chunk_training
        self.short_chunk_threshold = short_chunk_threshold
        self.short_chunk_size = short_chunk_size

        self.list_expand_left_layer_idx = []
        self.list_num_left_chunks = []
        for layer_idx in range(0, num_encoder_layers, expand_period):
            self.list_expand_left_layer_idx.append(layer_idx)
            self.list_num_left_chunks.append(num_left_chunks)
            num_left_chunks *= 2

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        encoder_layer = ConformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            layer_dropout,
            cnn_module_kernel,
            causal,
        )
        self.encoder = ConformerEncoder(encoder_layer, num_encoder_layers)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, warmup: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, d_model)
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        # Caution: We assume the subsampling factor is 4!

        #  lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1

        assert x.size(0) == lengths.max().item()

        src_key_padding_mask = make_pad_mask(lengths)

        max_len = x.size(0)
        chunk_size = torch.randint(1, max_len, (1,)).item()
        if chunk_size > (max_len * self.short_chunk_threshold):
            # full context
            chunk_size = max_len
        else:
            chunk_size = chunk_size % self.short_chunk_size + 1

        attn_masks = []
        for num_left_chunks in self.list_num_left_chunks:
            mask = ~subsequent_chunk_mask(
                size=x.size(0),
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks,
                device=x.device,
            )
            attn_masks.append(mask)

        x, _, _ = self.encoder(
            x,
            pos_emb,
            attn_masks=attn_masks,
            src_key_padding_mask=src_key_padding_mask,
            warmup=warmup,
        )  # (T, N, C)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        warmup: float = 1.0,
        states: Optional[Tensor] = None,
        chunk_size: int = 16,
        left_context: int = 64,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.
          states:
            The decode states for previous frames which contains the cached data.
            It has a shape of (2, encoder_layers, left_context, batch, attention_dim),
            states[0,...] is the attn_cache, states[1,...] is the conv_cache.
          chunk_size:
            The chunk size for decoding, this will be used to simulate streaming
            decoding using masking.
          left_context:
            How many old frames the attention can see in current chunk, it MUST
            be equal to left_context in decode_states.
          simulate_streaming:
            If setting True, it will use a masking strategy to simulate streaming
            fashion (i.e. every chunk data only see limited left context and
            right context). The whole sequence is supposed to be send at a time
            When using simulate_streaming.
        Returns:
          Return a tuple containing 2 tensors:
            - logits, its shape is (batch_size, output_seq_len, output_dim)
            - logit_lens, a tensor of shape (batch_size,) containing the number
              of frames in `logits` before padding.
            - decode_states, the updated DecodeStates including the information
              of current chunk.
        """

        # x: [N, T, C]
        # Caution: We assume the subsampling factor is 4!
        lengths = ((x_lens - 1) // 2 - 1) // 2

        if not simulate_streaming:
            assert (
                states is not None
            ), "Require cache when sending data in streaming mode"

            assert states.shape == (
                2,
                self.encoder_layers,
                left_context,
                x.size(0),
                self.d_model,
            ), f"""The shape of states MUST be equal to
             (2, encoder_layers, left_context, batch, d_model) which is
             {(2, self.encoder_layers, left_context, x.size(0), self.d_model)}
             given {states.shape}."""

            src_key_padding_mask = make_pad_mask(lengths + left_context)

            embed = self.encoder_embed(x)
            embed, pos_enc = self.encoder_pos(embed, left_context)
            embed = embed.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)

            x, states = self.encoder(
                embed,
                pos_enc,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
                states=states,
                left_context=left_context,
            )  # (T, B, F)

        else:
            assert states is None

            src_key_padding_mask = make_pad_mask(lengths)
            x = self.encoder_embed(x)
            x, pos_emb = self.encoder_pos(x)
            x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

            assert x.size(0) == lengths.max().item()

            num_left_chunks = -1
            if left_context >= 0:
                assert left_context % chunk_size == 0
                num_left_chunks = left_context // chunk_size

            mask = ~subsequent_chunk_mask(
                size=x.size(0),
                chunk_size=chunk_size,
                num_left_chunks=num_left_chunks,
                device=x.device,
            )
            x, _ = self.encoder(
                x,
                pos_emb,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
            )  # (T, N, C)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return x, lengths, states

    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        states: List[Tensor],
        left_context_size: int,
    ):
        """This is for real streaming decoding.
        Args:
          x:
            Input tensor with shape of (batch_size, chunk_size, feature_dim).
          x_lens:
            A tensor of shape (batch,) containing the number of frames in
            `x` before padding.
          states:
            Cached states including:
            - real lengths of cached attention left context of each sample in
              batch, with shape of (batch_size,).
            - attention caches, with shape of
              (num_encoder_layers, 2, left_context_length, batch_size, d_model).
            - convolution caches, with shape of
              (num_encoder_layers, batch_size, d_model, cnn_module_kernel - 1).
          left_context_size:
            Length of left context frames.

        Returns:
        """
        batch_size, chunk_size, _ = x.size()

        assert (
            states is not None
        ), "Require states when sending data in streaming mode"
        assert len(states) == 3, len(states)

        cached_left_context_sizes = states[0]
        assert cached_left_context_sizes.shape == (
            batch_size,
        ), cached_left_context_sizes.shape

        attn_caches = states[1]
        assert attn_caches.shape == (
            self.num_encoder_layers,
            2,
            left_context_size,
            batch_size,
            self.d_model,
        ), attn_caches.shape

        conv_caches = states[2]
        assert conv_caches.shape == (
            self.num_encoder_layers,
            batch_size,
            self.d_model,
            self.cnn_module_kernel - 1,
        ), conv_caches.shape

        x = self.encoder_embed(x)
        # query: [chunk] -> key: [left context, chunk]
        # relative distance of index i in query and index j in key is in range:
        # [-(chunk_size - 1), left_context_size + chunk_size - 1]
        x, pos_emb = self.encoder_pos(
            x, pos_len=chunk_size + left_context_size, neg_len=chunk_size
        )
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        # Caution: We assume the subsampling factor is 4!
        # lengths = ((x_lens - 1) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        lengths = (((x_lens - 1) >> 1) - 1) >> 1

        padding_mask = make_pad_mask(lengths + left_context_size)
        attn_mask = ~chunk_mask_with_left_context(
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            cached_left_context_sizes=cached_left_context_sizes,
            device=x.device,
        )

        x, new_attn_caches, new_conv_caches = self.encoder(
            x,
            pos_emb,
            attn_mask=attn_mask,
            padding_mask=padding_mask,
            attn_caches=attn_caches,
            conv_caches=conv_caches,
            attn_left_context_length=left_context_length,
        )  # (T, B, F)

        x = x.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)

        # update
        cached_left_context_sizes += chunk_size
        cached_left_context_sizes.clamp_(max=left_context_size)

        return x, new_attn_caches, new_conv_caches


class ConformerEncoderLayer(nn.Module):
    """ConformerEncoderLayer is made up of self-attn, feedforward and
    convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Args:
      d_model:
        The number of expected features in the input (required).
      nhead:
        The number of heads in the multiheadattention models (required).
      dim_feedforward:
        The dimension of the feedforward network model (default=2048).
      dropout:
        The dropout value (default=0.1).
      cnn_module_kernel (int):
        Kernel size of convolution module.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 31,
    ) -> None:
        super(ConformerEncoderLayer, self).__init__()

        self.layer_dropout = layer_dropout

        self.d_model = d_model

        self.self_attn = RelPositionMultiheadAttention(
            d_model, nhead, dropout=0.0
        )

        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.feed_forward_macaron = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel)

        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median) # noqa
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        warmup: float = 1.0,
        attn_cache: Optional[Tensor] = None,
        conv_cache: Optional[Tensor] = None,
        left_context_length: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Pass the input through the encoder layer.

        Args:
          src (Tensor):
            Input sequence to the encoder layer with shape (L, N, E).
          pos_emb (Tensor):
            Positional embedding tensor of shape (N, PE, E).
            In streaming decoding mode, PE = left_context_length + 2 * L - 1;
            otherwise, PE = 2 * L - 1.
          attn_mask (Tensor, optional):
            The attention mask for streaming simulation.
            Its shape is (L, L).
          src_key_padding_mask (Tensor, optional):
            The mask for the src keys per batch.
            In streaming decoding mode, its shape is (N, left_context_length + L). # noqa
            otherwise, its shape is (N, L).
          warmup (float, optional):
            It controls selective bypass of of layers;
            if < 1.0, we will bypass layers more frequently.
          attn_cache (Tensor, optional):
            Cached attention key and value tensors of left context frames.
            Its shape is (2, left_context_length, N, E).
            cache[0, ...] is key and cache[1, ...] is value.
          conv_cache (Tensor, optional):
            Cached left context frames for causal convolution.
            Its shape is (N, E, kernel_size - 1).
          left_context_length (Tensor, optional):
            Length of cached left context frames for attention.
            It is used only in real streaming decoding;
            in other circumstances, it MUST be 0.

          L: length of input sequence;
          N: batch size;
          E: feature number.

        Returns:
          A tuple of 3 tensors:
            - output sequence, with shape of (L, N, E)
            - updated attention cache,
              with shape of (2, left_context_length, N, E)
            - updated convolution cache,
              with shape of (N, E, kernel_size - 1)
        """
        src_orig = src

        warmup_scale = min(0.1 + warmup, 1.0)
        # alpha = 1.0 means fully use this encoder layer, 0.0 would mean
        # completely bypass it.
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else 0.1
            )
        else:
            alpha = 1.0

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # multi-headed self-attention module
        src_attn, attn_cache, _ = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            left_context_length=left_context_length,
        )
        src = src + self.dropout(src_attn)

        # convolution module
        src_conv, conv_cache = self.conv_module(src, conv_cache)
        src = src + self.dropout(src_conv)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        return src, attn_cache, conv_cache


class ConformerEncoder(nn.Module):
    r"""ConformerEncoder is a stack of N encoder layers

    Args:
      encoder_layer:
        An instance of the ConformerEncoderLayer() class.
      num_layers:
        The number of sub-encoder-layers in the encoder.

    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        expand_layer_indexes: Optional[Set[int]] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers
        # e.g., [6, 9], indicates the attention left context length will be
        # expanded at layer-6 and layer-9
        self.expand_layer_indexes = expand_layer_indexes

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        attn_masks: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        warmup: float = 1.0,
        attn_caches: Optional[Tensor] = None,
        conv_caches: Optional[Tensor] = None,
        attn_left_context_lengths: Optional[List[int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
          src (Tensor):
            Input sequence to the encoder layer with shape (L, N, E).
          pos_emb (Tensor):
            Positional embedding tensor of shape (N, PE, E).
            In streaming decoding mode, PE = left_context_length + 2 * L - 1;
            otherwise, PE = 2 * L - 1.
          attn_masks (Tensor, optional):
            List of attention masks for streaming simulation.
            The shape of each element is (L, L).
          src_key_padding_mask (Tensor, optional):
            The mask for the src keys per batch.
            In streaming decoding mode, its shape is (N, left_context_length + L). # noqa
            otherwise, its shape is (N, L).
          warmup (float, optional):
            It controls selective bypass of of layers;
            if < 1.0, we will bypass layers more frequently.
          attn_caches (Tensor, optional):
            Cached attention key and value tensors of left context frames.
            Its shape is (num_encoder_layers, 2, left_context_length, N, E).
            cache[0, ...] is key and cache[1, ...] is value.
          conv_caches (Tensor, optional):
            Cached left context frames for causal convolution.
            Its shape is (num_encoder_layers, N, E, kernel_size - 1).
          attn_left_context_lengths (Tensor, optional):
            List of lengths of cached left context frames for attention.
            It is used only in real streaming decoding;
            in other circumstances, it MUST be None.

        Shape:
            src: (S, N, E).
            pos_emb: (N, 2*S-1, E), for streaming decoding it is (N, 2*(S+left_context)-1, E).
            mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number

        """
        streaming_mode = True if attn_masks is None else False

        expand_idx = -1
        expand_layer_indexes: List[int] = []
        if self.expand_layer_indexes is not None:
            expand_layer_indexes = sorted(list(self.expand_layer_indexes))
            if not streaming_mode:
                assert len(attn_masks) - 1 == len(expand_layer_indexes)
            else:
                assert len(attn_left_context_lengths) - 1 == len(
                    expand_layer_indexes
                )
            expand_idx = expand_layer_indexes.pop(0)

        if not streaming_mode:
            assert attn_caches is None
            assert conv_caches is None
            assert attn_left_context_lengths is None
            mask = attn_masks[0]
            attn_left = 0
        else:
            mask = None
            attn_left = attn_left_context_lengths[0]

        new_attn_caches = []
        new_conv_caches = []
        output = src
        for layer_idx, mod in enumerate(self.layers):
            if layer_idx == expand_idx:
                if not streaming_mode:
                    mask = attn_masks[expand_idx]
                else:
                    attn_left = attn_left_context_lengths[expand_idx]
                if len(expand_layer_indexes) > 0:
                    expand_idx = expand_layer_indexes.pop(0)

            output, attn_cache, conv_cache = mod(
                output,
                pos_emb,
                attn_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
                attn_cache=attn_caches[layer_idx] if streaming_mode else None,
                conv_caches=conv_caches[layer_idx] if streaming_mode else None,
                attn_left_context_length=attn_left,
            )
            new_attn_caches.append(attn_cache)
            new_conv_caches.append(conv_cache)

        new_attn_caches = torch.cat(new_attn_caches)
        new_conv_caches = torch.cat(new_conv_caches)
        return output, new_attn_caches, new_conv_caches


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py  # noqa

    Suppose:
      i -> position of query,
      j -> position of key(value),
    we use positive relative position embedding when key(value) is to the
    left of query(i.e., i > j) and negative embedding otherwise.

    Args:
        d_model: Embedding dimension.
        dropout: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(
        self, d_model: int, dropout: float, max_len: int = 5000
    ) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = None
        self.pos_len = max_len
        self.neg_len = max_len
        self.gen_pe_positive()
        self.gen_pe_negative()

    def gen_pe_positive(self) -> None:
        """Generate the positive positional encodings."""
        pe_positive = torch.zeros(self.pos_len, self.d_model)
        position_positive = torch.arange(
            0, self.pos_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position_positive * div_term)
        pe_positive[:, 1::2] = torch.cos(position_positive * div_term)
        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
        self.pe_positive = torch.flip(pe_positive, [0])

    def gen_pe_negative(self) -> None:
        """Generate the negative positional encodings."""
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_negative = torch.zeros(self.neg_len, self.d_model)
        position_negative = torch.arange(
            0, self.neg_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_negative[:, 0::2] = torch.sin(-1 * position_negative * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position_negative * div_term)
        self.pe_negative = pe_negative

    def get_pe(
        self,
        pos_len: int,
        neg_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get positional encoding given positive length and negative length."""
        if self.pe_positive.dtype != dtype or str(
            self.pe_positive.device
        ) != str(device):
            self.pe_positive = self.pe_positive.to(dtype=dtype, device=device)
        if self.pe_negative.dtype != dtype or str(
            self.pe_negative.device
        ) != str(device):
            self.pe_negative = self.pe_negative.to(dtype=dtype, device=device)
        pe = torch.cat(
            [
                self.pe_positive[self.pos_len - pos_len :],
                self.pe_negative[1:neg_len],
            ],
            dim=0,
        )
        return pe

    def forward(
        self,
        x: torch.Tensor,
        pos_len: int,
        neg_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale input x and get positional encoding.
        Args:
            x (torch.Tensor): Input tensor (`*`).

        Returns:
          torch.Tensor:
            Encoded tensor of shape (`*`).
          torch.Tensor:
            Position embedding of shape (pos_len + neg_len - 1, `*`).
        """
        if pos_len > self.pos_len:
            self.pos_len = pos_len
            self.gen_pe_positive()
        if neg_len > self.neg_len:
            self.neg_len = neg_len
            self.gen_pe_negative()
        pos_emb = self.get_pe(pos_len, neg_len, x.device, x.dtype)
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionMultiheadAttention(nn.Module):
    r"""Multi-Head Attention layer with relative position encoding

    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.

    Examples::

        >>> rel_pos_multihead_attn = RelPositionMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super(RelPositionMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = ScaledLinear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = ScaledLinear(
            embed_dim, embed_dim, bias=True, initial_scale=0.25
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_u_scale = nn.Parameter(torch.zeros(()).detach())
        self.pos_bias_v_scale = nn.Parameter(torch.zeros(()).detach())
        self._reset_parameters()

    def _pos_bias_u(self):
        return self.pos_bias_u * self.pos_bias_u_scale.exp()

    def _pos_bias_v(self):
        return self.pos_bias_v * self.pos_bias_v_scale.exp()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_bias_u, std=0.01)
        nn.init.normal_(self.pos_bias_v, std=0.01)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        left_context: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
            pos_emb: Positional embedding tensor
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            left_context (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            pos_emb,
            self.embed_dim,
            self.num_heads,
            self.in_proj.get_weight(),
            self.in_proj.get_bias(),
            self.dropout,
            self.out_proj.get_weight(),
            self.out_proj.get_bias(),
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            left_context=left_context,
        )

    def rel_shift(self, x: Tensor, left_context_length: int = 0) -> Tensor:
        """Compute relative positional encoding.

        Args:
          x:
            Input tensor of shape (batch, nhead, tgt_len, PE),
            where src_len is the length of query tensor.
            For streaming decoding, PE = left_context + 2 * tgt_len - 1;
            otherwise, PE = 2 * tgt_len - 1
          left_context_length:
            Length of cached left context frames for attention.
            It is used only in real streaming decoding;
            in other circumstances, it MUST be 0.


        Returns:
            A tensor of shape (batch, nhead, tgt_len, src_len),
            where src_len is the length of key tensor.
        """
        (batch_size, num_heads, tgt_len, PE) = x.shape

        src_len = tgt_len + left_context_length
        assert src_len == PE - (tgt_len - 1)

        # Note: TorchScript requires explicit arg for stride()
        batch_stride = x.stride(0)
        nhead_stride = x.stride(1)
        tgt_len_stride = x.stride(2)
        PE_stride = x.stride(3)
        return x.as_strided(
            (batch_size, num_heads, tgt_len, src_len),
            (batch_stride, nhead_stride, tgt_len_stride - PE_stride, PE_stride),
            storage_offset=PE_stride * (tgt_len - 1),
        )

    def multi_head_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_emb: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        left_context_length: int = 0,
        cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
          query, key, value: map a query and a set of key-value pairs to an output.
          pos_emb: Positional embedding tensor
          embed_dim_to_check: total dimension of the model.
          num_heads: parallel attention heads.
          in_proj_weight, in_proj_bias: input projection weight and bias.
          dropout_p: probability of an element to be zeroed.
          out_proj_weight, out_proj_bias: the output projection weight and bias.
          training: apply dropout if is ``True``.
          key_padding_mask: if provided, specified padding elements in the key will
              be ignored by the attention. This is an binary mask. When the value is True,
              the corresponding value on the attention layer will be filled with -inf.
          need_weights: output attn_output_weights.
          attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
              the batches while a 3D mask allows to specify a different mask for the entries of each batch.
          left_context_length (int):
            Length of left context (in frames) used during streaming decoding.
            It is used only in real streaming decoding;
            in other circumstances, it MUST be 0.
          cache (tensor):
            Cached key and value tensors of left context frames.
            Its shape is (2, left_context_length, batch, nhead, head_dim).
            cache[0, ...] is key and cache[1, ...] is value.

        Shape:
            Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

            Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = nn.functional.linear(
                query, in_proj_weight, in_proj_bias
            ).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError(
                        "The size of the 2D attn_mask is not correct."
                    )
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError(
                        "The size of the 3D attn_mask is not correct."
                    )
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()
                    )
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if (
            key_padding_mask is not None
            and key_padding_mask.dtype == torch.uint8
        ):
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = (q * scaling).contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(src_len, bsz, num_heads, head_dim)
        v = v.contiguous().view(src_len, bsz, num_heads, head_dim)

        if cache is not None:
            # pad cached key and value of left context
            key_cache = cache[0]
            val_cache = cache[1]
            k = torch.cat([key_cache, k], dim=0)
            v = torch.cat([val_cache, v], dim=0)
            src_len = k.size(0)
        # update attention cache
        new_key_cache = k[src_len - left_context_length :]
        new_val_cache = v[src_len - left_context_length :]
        new_cache = torch.stack([new_key_cache, new_val_cache], dim=0)

        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == src_len, "{} == {}".format(
                key_padding_mask.size(1), src_len
            )

        q = q.transpose(0, 1)  # (batch, time1, head, d_k)

        pos_emb_bsz = pos_emb.size(0)
        assert pos_emb_bsz in (1, bsz)  # actually it is 1
        p = self.linear_pos(pos_emb).view(pos_emb_bsz, -1, num_heads, head_dim)
        # (batch, 2*time1, head, d_k) --> (batch, head, d_k, 2*time -1)
        p = p.permute(0, 2, 3, 1)

        q_with_bias_u = (q + self._pos_bias_u()).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        q_with_bias_v = (q + self._pos_bias_v()).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)
        matrix_ac = torch.matmul(
            q_with_bias_u, k
        )  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(
            q_with_bias_v, p
        )  # (batch, head, time1, 2*time1-1)
        matrix_bd = self.rel_shift(matrix_bd, left_context_length)

        attn_output_weights = (
            matrix_ac + matrix_bd
        )  # (batch, head, time1, time2)

        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, -1
        )

        assert list(attn_output_weights.size()) == [
            bsz * num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)

        # If we are using dynamic_chunk_training and setting a limited
        # num_left_chunks, the attention may only see the padding values which
        # will also be masked out by `key_padding_mask`, at this circumstances,
        # the whole column of `attn_output_weights` will be `-inf`
        # (i.e. be `nan` after softmax), so, we fill `0.0` at the masking
        # positions to avoid invalid loss value below.
        if (
            attn_mask is not None
            and attn_mask.dtype == torch.bool
            and key_padding_mask is not None
        ):
            combined_mask = attn_mask.unsqueeze(0) | key_padding_mask.unsqueeze(
                1
            ).unsqueeze(2)
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                combined_mask, 0.0
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(
            attn_output, out_proj_weight, out_proj_bias
        )

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
            return attn_output, new_cache, attn_output_weights
        else:
            return attn_output, new_cache, None


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py  # noqa

    Args:
      channels (int):
        The number of channels of conv layers.
      kernel_size (int):
        Kernerl size of conv layers.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = ScaledConv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # After pointwise_conv1 we put x through a gated linear unit
        # (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in
        # the range 1 to 4, but sometimes, for some reason, for layer 0 the rms
        # ends up being very large, between 50 and 100 for different channels.
        # This will cause very peaky and sparse derivatives for the sigmoid
        # gating function, which will tend to make the loss function not learn
        # effectively.  (for most layers the average absolute values are in the
        # range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for
        # different layers, which likely breaks down as 0.5 for the "linear"
        # half and 0.2 to 0.3 for the part that goes into the sigmoid.
        # The idea is that if we constrain the rms values to a reasonable range
        # via a constraint of max_abs=10.0, it will be in a better position to
        # start learning something, i.e. to latch onto the correct range.
        self.deriv_balancer1 = ActivationBalancer(
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
        )

        # make it causal by padding cached (kernel_size - 1) frames on the left
        self.cache_size = kernel_size - 1
        self.depthwise_conv = ScaledConv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=bias,
        )

        self.deriv_balancer2 = ActivationBalancer(
            channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.25,
        )

    def forward(self, x: Tensor, cache: Optional[Tensor] = None) -> Tensor:
        """Causal convolution module.

        Args:
          x (Tensor):
            Input tensor of shape (time, batch, channel).
          cache (Tensor):
            Cached left context for causal convolution.
            Its shape is (batch, channel, cache_size).

        Returns:
          A tensor of shape (time, batch, channels).
        """
        time, batch, channel = x.size()
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (batch, channels, time).

        # point-wise conv and GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2 * channel, time)
        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, time)

        if cache is None:
            cache = torch.zeros(
                batch, channel, self.cache_size, device=x.device, dtype=x.dtype
            )
        else:
            assert cache.shape == (batch, channel, self.cache_size)
        # make depthwise_conv causal by manualy padding cache tensor to the left
        x = torch.cat([cache, x], dim=2)
        # update cache
        new_cache = x[:, :, -self.cache_size :]

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        # point-wise conv
        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1), new_cache


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

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
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
        """
        assert in_channels >= 7
        super().__init__()

        self.conv = nn.Sequential(
            ScaledConv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=1,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
            ScaledConv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            DoubleSwish(),
        )
        self.out = ScaledLinear(
            layer3_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels
        )
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.
        self.out_norm = BasicNorm(out_channels, learn_eps=False)
        # constrain median of output to be close to zero.
        self.out_balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55
        )

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
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x = self.out_norm(x)
        x = self.out_balancer(x)
        return x


if __name__ == "__main__":
    feature_dim = 50
    c = Conformer(num_features=feature_dim, d_model=128, nhead=4)
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
        warmup=0.5,
    )
