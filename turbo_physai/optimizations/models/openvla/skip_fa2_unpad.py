# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
OpenVLA skip-FA2-unpad replacement for TurboPhysAI.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Callable, Optional


def make_fast_fa2_causal_mask_wrapper(
    original: Callable, options: Optional[Mapping[str, Any]] = None
) -> Callable:
    """Wrapper factory ``(original, options) -> wrapped _update_causal_mask``.

    Enabling the Group installs the returned callable in place of
    ``LlamaModel._update_causal_mask``.  ``options`` (Group options) is accepted
    for framework ``wrap`` compatibility and intentionally unused.
    """

    del options

    @functools.wraps(original)
    def wrapper(self, attention_mask, input_tensor, cache_position, past_seen_tokens):
        # Prefill (no KV cache) + FA2 + an explicit mask: skip the causal-mask
        # materialisation entirely so FA2 never enters the varlen/unpad path.
        if (
            self.config._attn_implementation == "flash_attention_2"
            and attention_mask is not None
            and past_seen_tokens == 0
        ):
            return None
        return original(self, attention_mask, input_tensor, cache_position, past_seen_tokens)

    return wrapper
