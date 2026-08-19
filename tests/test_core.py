from __future__ import annotations

import pytest
import torch

from paged_precision.policies import AttentionEMAPolicy, RecentPolicy, ResidencyObservation, SinkPolicy
from paged_precision.quantization import DirectKVQuantizerPair, KVQuantizerPair
from paged_precision.runtime import DirectPackedCacheLayout, PackedCacheLayout, PackedDirectKVCache, PagedPrecisionCache
from paged_precision.runtime.attention import packed_decode_attention


def test_base_plus_residual_exactly_reproduces_the_full_codebook() -> None:
    pair = KVQuantizerPair.for_head_dim(8, seed=7).to("cpu", torch.bfloat16)
    vectors = torch.randn(1, 2, 8, 8, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    encoding = pair.key.encode(vectors)

    full = pair.key.decode_full(encoding)
    direct = pair.key.decode_direct(vectors)
    assert torch.equal(full, direct)
    assert not torch.equal(pair.key.decode_base(encoding), full)


def test_packed_memory_moves_residuals_between_hbm_and_dram() -> None:
    pair = KVQuantizerPair.for_head_dim(8, seed=11)
    keys = torch.randn(1, 2, 8, 8, generator=torch.Generator().manual_seed(11))
    values = torch.randn(1, 2, 8, 8, generator=torch.Generator().manual_seed(12))
    snapshots = []

    for fraction in (0.0, 0.5, 1.0):
        cache = PagedPrecisionCache(
            PackedCacheLayout(
                layers=1,
                batch_size=1,
                kv_heads=2,
                head_dim=8,
                max_tokens=8,
                hot_fraction=fraction,
                block_size=4,
                norm_dtype=torch.float32,
            ),
            device="cpu",
        )
        cache.prepare_prefill(8, hot_block_ids=[] if fraction == 0 else None)
        key_encoding = pair.key.encode(keys)
        cache.store_prefill_layer(0, key_encoding, pair.value.encode(values))
        snapshots.append(cache.memory_snapshot())
        if fraction == 1:
            full_decode = cache.decode_rotated_blocks(0, 2, 0, "key", pair.key, dtype=torch.float32)
            expected = pair.key.decode_rotated(key_encoding, refined=True, dtype=torch.float32)
            assert torch.allclose(full_decode, expected, atol=1e-6, rtol=1e-6)

    assert [row.cache_hbm_bytes for row in snapshots] == sorted(row.cache_hbm_bytes for row in snapshots)
    assert snapshots[0].pinned_dram_bytes == 0
    assert snapshots[1].pinned_dram_bytes > 0
    assert snapshots[2].pinned_dram_bytes == 0
    assert snapshots[1].cache_hbm_bytes + snapshots[1].pinned_dram_bytes == snapshots[2].cache_hbm_bytes


def test_attention_ema_respects_its_capacity() -> None:
    policy = AttentionEMAPolicy(hot_fraction=0.375, sink_blocks=1, recent_blocks=1)
    plan = policy.observe(
        ResidencyObservation(
            visible_blocks=8,
            hot_blocks=torch.tensor([True, True, False, False, False, False, False, False]),
            previous_attention_mass=torch.tensor([0.01, 0.02, 0.80, 0.05, 0.04, 0.03, 0.02, 0.03]),
            hot_capacity=3,
        )
    )

    assert plan.selected == (0, 2, 7)
    assert plan.promote == (2, 7)
    assert plan.demote == (1,)
    assert policy.retention_mode == "recoverable"
    assert not policy.admit_new_blocks


def test_sink_and_recent_policies_select_opposite_ends() -> None:
    observation = ResidencyObservation(
        visible_blocks=8,
        hot_blocks=torch.tensor([False, False, False, False, False, True, True, True]),
        previous_attention_mass=None,
        hot_capacity=3,
    )

    sink = SinkPolicy(0.25)
    recent = RecentPolicy(0.25)
    assert sink.observe(observation).selected == (0, 1, 2)
    assert recent.observe(observation).selected == (5, 6, 7)
    assert sink.retention_mode == recent.retention_mode == "discardable"
    assert not sink.admit_new_blocks
    assert recent.admit_new_blocks


def test_direct_three_bit_cache_matches_dense_gqa_attention() -> None:
    pair = DirectKVQuantizerPair.for_head_dim(8, bits=3, seed=17).to("cpu", torch.float32)
    keys = torch.randn(1, 2, 6, 8, generator=torch.Generator().manual_seed(17))
    values = torch.randn(1, 2, 6, 8, generator=torch.Generator().manual_seed(18))
    key_encoding = pair.key.encode(keys)
    value_encoding = pair.value.encode(values)
    cache = PackedDirectKVCache(
        DirectPackedCacheLayout(1, 1, 2, 8, 8, 3, block_size=4, norm_dtype=torch.float32),
        device="cpu",
    )
    cache.prepare_prefill(6)
    cache.store_prefill_layer(0, key_encoding, value_encoding)
    decoded_keys = cache.decode_rotated_blocks(0, 2, 0, "key", pair.key, dtype=torch.float32)
    decoded_values = cache.decode_rotated_blocks(0, 2, 0, "value", pair.value, dtype=torch.float32)

    query = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(19))
    actual = packed_decode_attention(
        cache,
        query,
        layer=0,
        key_quantizer=pair.key,
        value_quantizer=pair.value,
    ).values
    rotated_query = query @ pair.key.rotation
    grouped_query = rotated_query.reshape(1, 2, 2, 8)
    logits = torch.einsum("bkgd,bktd->bkgt", grouped_query, decoded_keys) / (8**0.5)
    weights = torch.softmax(logits, dim=-1)
    rotated_output = torch.einsum("bkgt,bktd->bkgd", weights, decoded_values).reshape(1, 4, 8)
    expected = rotated_output @ pair.value.rotation.T

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_hot_set_transition_is_exact_and_reversible() -> None:
    cache = PagedPrecisionCache(
        PackedCacheLayout(1, 1, 1, 8, 16, 0.5, block_size=4, norm_dtype=torch.float32),
        device="cpu",
    )
    cache.prepare_prefill(16)
    pair = KVQuantizerPair.for_head_dim(8, seed=23)
    keys = torch.randn(1, 1, 16, 8, generator=torch.Generator().manual_seed(23))
    values = torch.randn(1, 1, 16, 8, generator=torch.Generator().manual_seed(24))
    cache.store_prefill_layer(0, pair.key.encode(keys), pair.value.encode(values))
    expected_dram = cache.memory_snapshot().pinned_dram_bytes
    assert expected_dram == 2 * cache.layout.layers * 2 * cache.layout.packed_block_bytes

    cache.set_hot_blocks([0, 2])
    assert {block for block in range(4) if cache.is_hot(block)} == {0, 2}
    assert cache.memory_snapshot().pinned_dram_bytes == expected_dram
    cache.set_hot_blocks([1, 3])
    assert {block for block in range(4) if cache.is_hot(block)} == {1, 3}
    assert cache.memory_snapshot().pinned_dram_bytes == expected_dram


def test_recoverable_new_block_does_not_displace_the_policy_hot_set() -> None:
    cache = PagedPrecisionCache(
        PackedCacheLayout(1, 1, 1, 8, 12, 0.5, block_size=4, norm_dtype=torch.float32),
        device="cpu",
    )
    cache.prepare_prefill(8, hot_block_ids=[0, 1])
    pair = KVQuantizerPair.for_head_dim(8, seed=29)
    keys = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(29))
    values = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(30))
    cache.store_prefill_layer(0, pair.key.encode(keys), pair.value.encode(values))

    new_key = pair.key.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(31)))
    new_value = pair.value.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(32)))
    cache.append_layer_token(0, 8, new_key, new_value)

    assert {block for block in range(3) if cache.is_hot(block)} == {0, 1}
    assert 2 in cache.cold


@pytest.mark.parametrize(
    ("hot_blocks", "admit_hot", "expected_hot"),
    [
        ([0, 1], False, {0, 1}),
        ([0, 1], True, {1, 2}),
    ],
)
def test_discardable_sink_and_recent_use_no_dram_at_a_block_boundary(
    hot_blocks: list[int],
    admit_hot: bool,
    expected_hot: set[int],
) -> None:
    cache = PagedPrecisionCache(
        PackedCacheLayout(1, 1, 1, 8, 16, 0.5, block_size=4, norm_dtype=torch.float32),
        device="cpu",
    )
    cache.prepare_prefill(8, hot_block_ids=hot_blocks, retention_mode="discardable")
    pair = KVQuantizerPair.for_head_dim(8, seed=37)
    keys = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(37))
    values = torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(38))
    cache.store_prefill_layer(0, pair.key.encode(keys), pair.value.encode(values))

    new_key = pair.key.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(39)))
    new_value = pair.value.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(40)))
    cache.append_layer_token(
        0,
        8,
        new_key,
        new_value,
        retention_mode="discardable",
        admit_hot=admit_hot,
    )

    assert {block for block in range(3) if cache.is_hot(block)} == expected_hot
    assert not cache.cold
    assert cache.memory_snapshot().pinned_dram_bytes == 0


def test_discardable_prefill_drops_nonresident_refinements() -> None:
    cache = PagedPrecisionCache(
        PackedCacheLayout(1, 1, 1, 8, 16, 0.5, block_size=4, norm_dtype=torch.float32),
        device="cpu",
    )
    cache.prepare_prefill(12, hot_block_ids=[0, 1], retention_mode="discardable")
    pair = KVQuantizerPair.for_head_dim(8, seed=41)
    keys = torch.randn(1, 1, 12, 8, generator=torch.Generator().manual_seed(41))
    values = torch.randn(1, 1, 12, 8, generator=torch.Generator().manual_seed(42))
    cache.store_prefill_layer(0, pair.key.encode(keys), pair.value.encode(values))

    assert not cache.cold
    assert cache.memory_snapshot().pinned_dram_bytes == 0
    with pytest.raises(RuntimeError, match="discarded refinements cannot be promoted"):
        cache.set_hot_blocks([1, 2], retention_mode="discardable")


def test_sink_does_not_promote_a_discarded_block() -> None:
    cache = PagedPrecisionCache(
        PackedCacheLayout(1, 1, 1, 8, 16, 0.5, block_size=4, norm_dtype=torch.float32),
        device="cpu",
    )
    cache.prepare_prefill(12, hot_block_ids=[0, 1], retention_mode="discardable")
    pair = KVQuantizerPair.for_head_dim(8, seed=43)
    keys = torch.randn(1, 1, 12, 8, generator=torch.Generator().manual_seed(43))
    values = torch.randn(1, 1, 12, 8, generator=torch.Generator().manual_seed(44))
    cache.store_prefill_layer(0, pair.key.encode(keys), pair.value.encode(values))
    new_key = pair.key.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(45)))
    new_value = pair.value.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(46)))
    cache.append_layer_token(
        0,
        12,
        new_key,
        new_value,
        retention_mode="discardable",
        admit_hot=False,
    )

    policy = SinkPolicy(0.5)
    plan = policy.observe(
        ResidencyObservation(
            visible_blocks=cache.available_blocks(),
            hot_blocks=cache.block_to_slot >= 0,
            previous_attention_mass=None,
            hot_capacity=cache.layout.hot_slots,
        )
    )
    assert plan.promote == ()
    cache.set_hot_blocks(list(plan.selected), retention_mode=policy.retention_mode)
    assert {block for block in range(4) if cache.is_hot(block)} == {0, 1}
    assert not cache.cold


@pytest.mark.parametrize(
    ("initial_hot", "admit_hot", "expected_hot"),
    [
        ([0, 1], False, {0, 1}),
        ([1, 2], True, {2, 3}),
    ],
)
def test_discarding_cold_refinements_preserves_the_decoded_cache(
    initial_hot: list[int],
    admit_hot: bool,
    expected_hot: set[int],
) -> None:
    pair = KVQuantizerPair.for_head_dim(8, seed=43)
    keys = torch.randn(1, 1, 12, 8, generator=torch.Generator().manual_seed(43))
    values = torch.randn(1, 1, 12, 8, generator=torch.Generator().manual_seed(44))
    new_key = pair.key.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(45)))
    new_value = pair.value.encode(torch.randn(1, 1, 1, 8, generator=torch.Generator().manual_seed(46)))
    caches = []

    for retention_mode in ("recoverable", "discardable"):
        cache = PagedPrecisionCache(
            PackedCacheLayout(1, 1, 1, 8, 16, 0.5, block_size=4, norm_dtype=torch.float32),
            device="cpu",
        )
        cache.prepare_prefill(12, hot_block_ids=initial_hot, retention_mode=retention_mode)
        cache.store_prefill_layer(0, pair.key.encode(keys), pair.value.encode(values))
        cache.append_layer_token(
            0,
            12,
            new_key,
            new_value,
            retention_mode=retention_mode,
            admit_hot=admit_hot,
        )
        caches.append(cache)

    recoverable, discardable = caches
    assert {block for block in range(4) if recoverable.is_hot(block)} == expected_hot
    assert {block for block in range(4) if discardable.is_hot(block)} == expected_hot
    assert recoverable.memory_snapshot().pinned_dram_bytes > 0
    assert discardable.memory_snapshot().pinned_dram_bytes == 0
    for side, quantizer in (("key", pair.key), ("value", pair.value)):
        retained = recoverable.decode_rotated_blocks(0, 4, 0, side, quantizer, dtype=torch.float32)
        dropped = discardable.decode_rotated_blocks(0, 4, 0, side, quantizer, dtype=torch.float32)
        assert torch.equal(retained, dropped)
