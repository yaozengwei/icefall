import torch


def test_attention_forward():
    from conformer import RelPositionMultiheadAttention

    d_model, num_heads = 256, 4
    batch_size, tgt_len = 2, 10
    attention = RelPositionMultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
    )

    src = torch.randn(tgt_len, batch_size, d_model)
    pos_emb = torch.randn(1, 2 * tgt_len - 1, d_model)
    attn_mask = torch.zeros(tgt_len, tgt_len, dtype=torch.bool)

    (
        src_attn,
        _,
        _,
    ) = attention(src, src, src, pos_emb=pos_emb, attn_mask=attn_mask)

    assert src_attn.shape == (tgt_len, batch_size, d_model)


def test_attention_infer():
    from conformer import RelPositionMultiheadAttention

    d_model, num_heads = 256, 4
    batch_size, tgt_len = 2, 10
    left_context_size = 4
    attention = RelPositionMultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
    )
    src = torch.randn(tgt_len, batch_size, d_model)
    pos_emb = torch.randn(1, left_context_size + 2 * tgt_len - 1, d_model)
    attn_cache = torch.randn(2, left_context_size, batch_size, d_model)

    (src_attn, new_attn_cache, _,) = attention(
        src,
        src,
        src,
        pos_emb=pos_emb,
        left_context_size=left_context_size,
        cache=attn_cache,
    )

    assert src_attn.shape == (tgt_len, batch_size, d_model)
    assert new_attn_cache.shape == (2, left_context_size, batch_size, d_model)


def test_conformer_encoder_layer_forward():
    from conformer import ConformerEncoderLayer

    d_model, num_heads, cnn_module_kernel = 256, 4, 31
    batch_size, tgt_len = 2, 10
    layer = ConformerEncoderLayer(
        d_model=d_model, nhead=num_heads, cnn_module_kernel=cnn_module_kernel
    )
    src = torch.randn(tgt_len, batch_size, d_model)
    pos_emb = torch.randn(1, 2 * tgt_len - 1, d_model)
    attn_mask = torch.zeros(tgt_len, tgt_len, dtype=torch.bool)
    src, _, _ = layer(src, pos_emb=pos_emb, attn_mask=attn_mask)
    assert src.shape == (tgt_len, batch_size, d_model)


def test_conformer_encoder_layer_infer():
    from conformer import ConformerEncoderLayer

    d_model, num_heads, cnn_module_kernel = 256, 4, 31
    batch_size, tgt_len = 2, 10
    left_context_size = 4
    layer = ConformerEncoderLayer(
        d_model=d_model, nhead=num_heads, cnn_module_kernel=cnn_module_kernel
    )
    src = torch.randn(tgt_len, batch_size, d_model)
    pos_emb = torch.randn(1, left_context_size + 2 * tgt_len - 1, d_model)
    attn_cache = torch.randn(2, left_context_size, batch_size, d_model)
    conv_cache = torch.randn(batch_size, d_model, cnn_module_kernel - 1)
    src, new_attn_cache, new_conv_cache = layer(
        src,
        pos_emb=pos_emb,
        attn_cache=attn_cache,
        conv_cache=conv_cache,
        left_context_size=left_context_size,
    )
    assert src.shape == (tgt_len, batch_size, d_model)
    assert new_attn_cache.shape == (2, left_context_size, batch_size, d_model)
    assert new_conv_cache.shape == (batch_size, d_model, cnn_module_kernel - 1)


def test_conformer_encoder_forward():
    from conformer import ConformerEncoder, ConformerEncoderLayer

    d_model, num_heads, cnn_module_kernel, num_layers = 256, 4, 31, 2
    batch_size, tgt_len = 2, 10

    layer = ConformerEncoderLayer(
        d_model=d_model, nhead=num_heads, cnn_module_kernel=cnn_module_kernel
    )
    encoder = ConformerEncoder(layer, num_layers)
    src = torch.randn(tgt_len, batch_size, d_model)
    pos_emb = torch.randn(1, 2 * tgt_len - 1, d_model)
    attn_mask = torch.zeros(tgt_len, tgt_len, dtype=torch.bool)
    src, _, _ = encoder(src, pos_emb=pos_emb, attn_mask=attn_mask)
    assert src.shape == (tgt_len, batch_size, d_model)


def test_conformer_encoder_infer():
    from conformer import ConformerEncoder, ConformerEncoderLayer

    d_model, num_heads, cnn_module_kernel, num_layers = 256, 4, 31, 2
    batch_size, tgt_len = 2, 10
    left_context_size = 4

    layer = ConformerEncoderLayer(
        d_model=d_model, nhead=num_heads, cnn_module_kernel=cnn_module_kernel
    )
    encoder = ConformerEncoder(layer, num_layers)
    src = torch.randn(tgt_len, batch_size, d_model)
    pos_emb = torch.randn(1, left_context_size + 2 * tgt_len - 1, d_model)

    attn_caches = torch.randn(
        num_layers, 2, left_context_size, batch_size, d_model
    )
    conv_caches = torch.randn(
        num_layers,
        batch_size,
        d_model,
        cnn_module_kernel - 1,
    )
    src, new_attn_caches, new_conv_caches = encoder(
        src,
        pos_emb=pos_emb,
        attn_caches=attn_caches,
        conv_caches=conv_caches,
        left_context_size=left_context_size,
    )
    assert src.shape == (tgt_len, batch_size, d_model)
    assert new_attn_caches.shape == (
        num_layers,
        2,
        left_context_size,
        batch_size,
        d_model,
    )
    assert new_conv_caches.shape == (
        num_layers,
        batch_size,
        d_model,
        cnn_module_kernel - 1,
    )


def test_conformer_forward():
    from conformer import Conformer

    d_model, num_heads, cnn_module_kernel, num_layers = 256, 4, 31, 2
    num_features = 40
    num_left_chunks = -1
    batch_size, tgt_len = 2, 10
    input_len = tgt_len * 4 + 3
    model = Conformer(
        num_features=num_features,
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=num_layers,
        cnn_module_kernel=cnn_module_kernel,
        num_left_chunks=num_left_chunks,
    )

    x = torch.randn(batch_size, input_len, num_features)
    x_lens = torch.randint(1, input_len, (batch_size,))
    x_lens[0] = input_len
    x, out_x_lens = model(x, x_lens)
    assert x.shape == (batch_size, tgt_len, d_model)
    assert len(out_x_lens) == batch_size


def test_conformer_simulate_streaming_forward():
    from conformer import Conformer

    d_model, num_heads, cnn_module_kernel, num_layers = 256, 4, 31, 2
    num_features = 40
    batch_size, tgt_len = 2, 10
    input_len = tgt_len * 4 + 3
    chunk_size, left_context_size = 2, 2
    model = Conformer(
        num_features=num_features,
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=num_layers,
        cnn_module_kernel=cnn_module_kernel,
    )

    x = torch.randn(batch_size, input_len, num_features)
    x_lens = torch.randint(1, input_len, (batch_size,))
    x_lens[0] = input_len
    x, out_x_lens = model.simulate_streaming_forward(
        x, x_lens, chunk_size=chunk_size, left_context_size=left_context_size
    )
    assert x.shape == (batch_size, tgt_len, d_model)
    assert len(out_x_lens) == batch_size


def test_conformer_streaming_forward():
    from conformer import Conformer

    d_model, num_heads, cnn_module_kernel, num_layers = 256, 4, 31, 2
    num_features = 40
    batch_size, chunk_size = 2, 2
    input_len = chunk_size * 4 + 3
    left_context_size = 4
    model = Conformer(
        num_features=num_features,
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=num_layers,
        cnn_module_kernel=cnn_module_kernel,
    )

    attn_caches = torch.randn(
        num_layers, 2, left_context_size, batch_size, d_model
    )
    conv_caches = torch.randn(
        num_layers,
        batch_size,
        d_model,
        cnn_module_kernel - 1,
    )
    cached_left_context_sizes = torch.zeros(batch_size, dtype=torch.int)
    states = [cached_left_context_sizes, attn_caches, conv_caches]
    x = torch.randn(batch_size, input_len, num_features)
    x_lens = torch.randint(1, input_len, (batch_size,))
    x_lens[0] = input_len
    x, out_x_lens, new_states = model.streaming_forward(
        x,
        x_lens,
        states=states,
        chunk_size=chunk_size,
        left_context_size=left_context_size,
    )
    assert x.shape == (batch_size, chunk_size, d_model)
    assert len(out_x_lens) == batch_size
    assert len(new_states) == 3
    assert new_states[0].shape == (batch_size,)
    assert new_states[1].shape == (
        num_layers,
        2,
        left_context_size,
        batch_size,
        d_model,
    ), new_states[1].shape
    assert new_states[2].shape == (
        num_layers,
        batch_size,
        d_model,
        cnn_module_kernel - 1,
    )


def test_streaming_consistancy():
    # This is to check the consistancy between simulated streaming forward
    # and real streaming forward. To exclude the impact of feature subsampling,
    # we should first comment out the module `encoder_embed` in `Conformer` class. # noqa
    from conformer import Conformer

    d_model, num_heads, cnn_module_kernel, num_layers = 256, 4, 31, 2
    num_features = 40
    batch_size, chunk_size, num_chunks = 2, 2, 3
    left_context_size = 4
    model = Conformer(
        num_features=num_features,
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=num_layers,
        cnn_module_kernel=cnn_module_kernel,
    )
    model.eval()

    # simulate streaming forward
    input_len = num_chunks * chunk_size
    x = torch.randn(batch_size, input_len, d_model)
    x_lens = torch.ones(batch_size, dtype=torch.int) * input_len
    out_simulate, out_x_lens = model.simulate_streaming_forward(
        x, x_lens, chunk_size=chunk_size, left_context_size=left_context_size
    )

    # real streaming forward
    attn_caches = torch.zeros(
        num_layers, 2, left_context_size, batch_size, d_model
    )
    conv_caches = torch.zeros(
        num_layers,
        batch_size,
        d_model,
        cnn_module_kernel - 1,
    )
    cached_left_context_sizes = torch.zeros(batch_size, dtype=torch.int)
    states = [cached_left_context_sizes, attn_caches, conv_caches]
    for i in range(num_chunks):
        start = chunk_size * i
        end = start + chunk_size
        out_chunk, _, states = model.streaming_forward(
            x[:, start:end, :],
            torch.ones(batch_size, dtype=torch.int) * chunk_size,
            states=states,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
        )

        print(out_chunk - out_simulate[:, start:end, :])


if __name__ == "__main__":
    test_attention_forward()
    test_attention_infer()
    test_conformer_encoder_layer_forward()
    test_conformer_encoder_layer_infer()
    test_conformer_encoder_forward()
    test_conformer_encoder_infer()
    test_conformer_forward()
    test_conformer_simulate_streaming_forward()
    test_conformer_streaming_forward()
    # test_streaming_consistancy()
