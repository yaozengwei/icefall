import torch


def test_rel_positional_encoding():
    from emformer import RelPositionalEncoding

    D = 256
    pos_enc = RelPositionalEncoding(D, dropout=0.1)
    pos_len = 100
    neg_len = 100
    x = torch.randn(2, D)
    x, pos_emb = pos_enc(x, pos_len, neg_len)
    assert pos_emb.shape == (pos_len + neg_len - 1, D)


def test_emformer_attention_forward():
    from emformer import EmformerAttention

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 3
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    attention = EmformerAttention(embed_dim=D, nhead=8)

    for use_memory in [True, False]:
        if use_memory:
            S = num_chunks
            M = S - 1
        else:
            S, M = 0, 0

        Q, KV = R + U + S, M + R + U
        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        summary = torch.randn(S, B, D)
        memory = torch.randn(M, B, D)
        attention_mask = torch.rand(Q, KV) >= 0.5
        PE = 2 * U - 1
        pos_emb = torch.randn(PE, D)

        output_right_context_utterance, output_memory = attention(
            utterance,
            lengths,
            right_context,
            summary,
            memory,
            attention_mask,
            pos_emb,
        )
        assert output_right_context_utterance.shape == (R + U, B, D)
        assert output_memory.shape == (M, B, D)


def test_emformer_attention_infer():
    from emformer import EmformerAttention

    B, D = 2, 256
    U = 4
    R = 2
    L = 3
    attention = EmformerAttention(embed_dim=D, nhead=8)

    for use_memory in [True, False]:
        if use_memory:
            S, M = 1, 3
        else:
            S, M = 0, 0

        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        summary = torch.randn(S, B, D)
        memory = torch.randn(M, B, D)
        left_context_key = torch.randn(L, B, D)
        left_context_val = torch.randn(L, B, D)
        PE = L + 2 * U - 1
        pos_emb = torch.randn(PE, D)

        (
            output_right_context_utterance,
            output_memory,
            next_key,
            next_val,
        ) = attention.infer(
            utterance,
            lengths,
            right_context,
            summary,
            memory,
            left_context_key,
            left_context_val,
            pos_emb,
        )
        assert output_right_context_utterance.shape == (R + U, B, D)
        assert output_memory.shape == (S, B, D)
        assert next_key.shape == (L + U, B, D)
        assert next_val.shape == (L + U, B, D)


def test_convolution_module_forward():
    from emformer import ConvolutionModule

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 3
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    kernel_size = 31
    conv_module = ConvolutionModule(
        chunk_length, right_context_length, D, kernel_size,
    )

    utterance = torch.randn(U, B, D)
    right_context = torch.randn(R, B, D)
    cache = torch.randn(B, D, kernel_size - 1)

    utterance, right_context, new_cache = conv_module(
        utterance, right_context, cache
    )
    assert utterance.shape == (U, B, D)
    assert right_context.shape == (R, B, D)
    assert new_cache.shape == (B, D, kernel_size - 1)


def test_convolution_module_infer():
    from emformer import ConvolutionModule

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 1
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    kernel_size = 31
    conv_module = ConvolutionModule(
        chunk_length, right_context_length, D, kernel_size,
    )

    utterance = torch.randn(U, B, D)
    right_context = torch.randn(R, B, D)
    cache = torch.randn(B, D, kernel_size - 1)

    utterance, right_context, new_cache = conv_module.infer(
        utterance, right_context, cache
    )
    assert utterance.shape == (U, B, D)
    assert right_context.shape == (R, B, D)
    assert new_cache.shape == (B, D, kernel_size - 1)


def test_emformer_encoder_layer_forward():
    from emformer import EmformerEncoderLayer

    B, D = 2, 256
    chunk_length = 8
    right_context_length = 2
    left_context_length = 8
    kernel_size = 31
    num_chunks = 3
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length

    for use_memory in [True, False]:
        if use_memory:
            S = num_chunks
            M = S - 1
        else:
            S, M = 0, 0

        layer = EmformerEncoderLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=1024,
            chunk_length=chunk_length,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
        )

        Q, KV = R + U + S, M + R + U
        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        memory = torch.randn(M, B, D)
        attention_mask = torch.rand(Q, KV) >= 0.5
        PE = 2 * U - 1
        pos_emb = torch.randn(PE, D)

        output_utterance, output_right_context, output_memory = layer(
            utterance,
            lengths,
            right_context,
            memory,
            attention_mask,
            pos_emb,
        )
        assert output_utterance.shape == (U, B, D)
        assert output_right_context.shape == (R, B, D)
        assert output_memory.shape == (M, B, D)


def test_emformer_encoder_layer_infer():
    from emformer import EmformerEncoderLayer

    B, D = 2, 256
    chunk_length = 8
    right_context_length = 2
    left_context_length = 8
    kernel_size = 31
    num_chunks = 1
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length

    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0

        layer = EmformerEncoderLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=1024,
            chunk_length=chunk_length,
            cnn_module_kernel=kernel_size,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=M,
        )

        utterance = torch.randn(U, B, D)
        lengths = torch.randint(1, U + 1, (B,))
        lengths[0] = U
        right_context = torch.randn(R, B, D)
        memory = torch.randn(M, B, D)
        state = None
        PE = left_context_length + 2 * U - 1
        pos_emb = torch.randn(PE, D)
        conv_cache = None
        (
            output_utterance,
            output_right_context,
            output_memory,
            output_state,
            conv_cache,
        ) = layer.infer(
            utterance,
            lengths,
            right_context,
            memory,
            pos_emb,
            state,
            conv_cache,
        )
        assert output_utterance.shape == (U, B, D)
        assert output_right_context.shape == (R, B, D)
        if use_memory:
            assert output_memory.shape == (1, B, D)
        else:
            assert output_memory.shape == (0, B, D)
        assert len(output_state) == 4
        assert output_state[0].shape == (M, B, D)
        assert output_state[1].shape == (left_context_length, B, D)
        assert output_state[2].shape == (left_context_length, B, D)
        assert output_state[3].shape == (1, B)
        assert conv_cache.shape == (B, D, kernel_size - 1)


if __name__ == "__main__":
    test_rel_positional_encoding()
    test_emformer_attention_forward()
    test_emformer_attention_infer()
    test_convolution_module_forward()
    test_convolution_module_infer()
    test_emformer_encoder_layer_forward()
    test_emformer_encoder_layer_infer()
