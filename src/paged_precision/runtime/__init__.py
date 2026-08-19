"""Packed KV-cache storage and model integration."""

from paged_precision.runtime.adapters import DirectKVModelAdapter, PrefillBuildResult, PagedPrecisionModelAdapter, extract_legacy_cache
from paged_precision.runtime.cache import PackedCacheLayout, PagedPrecisionCache, RuntimeMemorySnapshot
from paged_precision.runtime.direct_cache import DirectPackedCacheLayout, PackedDirectKVCache
from paged_precision.runtime.packing import PackedIndices, pack_indices, packed_nbytes, unpack_indices

__all__ = [
    "PackedCacheLayout",
    "DirectPackedCacheLayout",
    "DirectKVModelAdapter",
    "PackedDirectKVCache",
    "PackedIndices",
    "PagedPrecisionCache",
    "PrefillBuildResult",
    "RuntimeMemorySnapshot",
    "PagedPrecisionModelAdapter",
    "extract_legacy_cache",
    "pack_indices",
    "packed_nbytes",
    "unpack_indices",
]
