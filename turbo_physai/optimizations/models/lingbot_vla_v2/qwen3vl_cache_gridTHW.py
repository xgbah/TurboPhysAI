# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Cache the Qwen3-VL vision position-embedding index/weight tables.

``Qwen3VLVisionModel.fast_pos_embed_interpolate`` is a pure function of
``grid_thw`` *up to* the ``self.pos_embed`` lookup.  The lookup itself cannot be
cached -- ``pos_embed`` is a trainable parameter, so its output changes every
step -- but everything around it can, and on the first step it costs a handful
of host-side ops (``linspace``, ``int``, ``tolist``) plus data-dependent scalars
that Dynamo cannot trace.

This module keeps the cached part and drops the compile path, which now lives in
``qwen3vl_compile_vision``.  The Group entry point is
:func:`fast_pos_embed_interpolate`, replaced onto
``lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla.Qwen3VLVisionModel``.

Cached per distinct ``grid_thw`` (keyed on its flattened values, plus device and
dtype):

* ``idx_tensor``  -- the four bilinear corner rows into ``pos_embed``
* ``weight_tensor`` -- the four bilinear weights
* ``split_sizes`` -- per-image ``h*w`` used to split the lookup result
* ``shapes`` -- ``(t, h/merge, w/merge, merge)`` used by the permute

The cache is bounded: when it reaches ``_POS_EMBED_CACHE_LIMIT`` entries it is
cleared rather than evicted one by one.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# fast_pos_embed_interpolate: cache the parameter-independent index tables
# ---------------------------------------------------------------------------
_POS_EMBED_CACHE: dict = {}
_POS_EMBED_CACHE_LIMIT = 16


def _build_pos_embed_tables(self, grid_thw: torch.Tensor, dtype: torch.dtype):
    """Build the position-embedding index/weight tables on the CPU.

    Mirrors ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` up to (but not
    including) the ``self.pos_embed`` lookup.  ``grid_thw`` is converted to
    Python ints once, which is what removes the data-dependent scalars.
    """
    rows = grid_thw.detach().to("cpu").tolist()
    num_grid_per_side = self.num_grid_per_side
    merge_size = self.config.spatial_merge_size

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for _t, h, w in rows:
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
    # Built in the target dtype straight from the Python floats, matching the
    # original ``torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype)``.
    weight_tensor = torch.tensor(weight_list, dtype=dtype)

    split_sizes = [int(h) * int(w) for _t, h, w in rows]
    shapes = [
        (int(t), int(h) // merge_size, int(w) // merge_size, merge_size)
        for t, h, w in rows
    ]
    return idx_tensor, weight_tensor, split_sizes, shapes


@torch._dynamo.disable
def _lookup_pos_embed_tables(self, grid_thw: torch.Tensor, device, dtype):
    """Return the cached tables for ``grid_thw``, building them on first sight."""
    key = (tuple(grid_thw.detach().to("cpu").flatten().tolist()), str(device), str(dtype))
    cached = _POS_EMBED_CACHE.get(key)
    if cached is None:
        idx_tensor, weight_tensor, split_sizes, shapes = _build_pos_embed_tables(
            self, grid_thw, dtype
        )
        if len(_POS_EMBED_CACHE) >= _POS_EMBED_CACHE_LIMIT:
            _POS_EMBED_CACHE.clear()
        # ``.to(device)`` only moves storage; it never changes the values.
        cached = (idx_tensor.to(device), weight_tensor.to(device), split_sizes, shapes)
        _POS_EMBED_CACHE[key] = cached
    return cached


def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Cached, traceable replacement for the Qwen3-VL vision position embedding."""
    weight = self.pos_embed.weight
    idx_tensor, weight_tensor, split_sizes, shapes = _lookup_pos_embed_tables(
        self, grid_thw, weight.device, weight.dtype
    )

    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split(split_sizes)

    patch_pos_embeds_permute = []
    for pos_embed, (t, h, w, merge_size) in zip(patch_pos_embeds, shapes):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h, merge_size, w, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    return torch.cat(patch_pos_embeds_permute)
