# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Make ``torch.compile`` effective for Qwen3-VL.

Two pieces that have to travel together:

* :func:`compile_vision_blocks` -- per-layer ``torch.compile`` of the vision
  blocks.  Split out of ``qwen3vl_cache_gridTHW`` (which only caches the
  position-embedding tables) so the compile path can be read and enabled alone.
* :class:`DisabledStepCounter` / :func:`wrap_init` -- keep ``FlowMatchingV2``'s
  dead ``self.steps`` counter out of Dynamo's guards.  Without this the forward
  recompiles every step and falls back to eager after eight steps, so the
  compiled vision blocks would stop being reached.

The Group entry points are :func:`compile_vision_blocks` (wrapped around
``lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla.forward_without_grid_thw``)
and :func:`wrap_init` (wrapped around
``FlowMatchingV2.__init__``).  ``options["compile_blocks"]`` switches the
compile path on and ``options["compile"]`` is forwarded verbatim to
``torch.compile``; with the switch off the original forward is returned
untouched.

Why compile inside the checkpoint
---------------------------------
``GradientCheckpointingLayer.__call__`` wraps the block computation with the
checkpoint HOP.  Compiling the block *object* would put ``torch.compile``
outside that HOP, where Dynamo breaks and the recomputed forward stays eager.
Compiling ``blk.forward`` and shadowing the bound method keeps ``__call__``'s
checkpoint branch, FSDP hooks and kwargs handling intact, so the recomputation
runs a compiled graph too.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch


def _compile_block_forward(blk, compile_options):
    """Compile one block's ``forward`` and let ``__call__`` reach it.

    The checkpoint wrapper lives in ``GradientCheckpointingLayer.__call__``
    (``self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)``),
    i.e. *outside* the block computation.  Compiling the block object would put
    ``torch.compile`` outside the checkpoint, where Dynamo breaks on the
    checkpoint HOP and the recomputed forward stays eager.  Instead we compile
    ``blk.forward`` and shadow the bound method on the instance, so the
    existing ``__call__`` keeps its checkpoint branch, FSDP hooks and kwargs
    handling, and the checkpoint recomputes a compiled graph.

    The compiled callable is cached on the instance; shadowing ``forward`` is
    idempotent across steps.
    """
    cached = getattr(blk, "_tp_compiled_forward", None)
    if cached is None:
        cached = torch.compile(blk.forward, **compile_options)
        blk._tp_compiled_forward = cached
        blk.forward = cached
    return cached


def compile_vision_blocks(original, options=None):
    """Group entry point: per-layer ``torch.compile`` for the vision blocks.

    ``options["compile_blocks"]`` enables the path; ``options["compile"]`` is
    forwarded verbatim to ``torch.compile`` (e.g. ``{"dynamic": True}``).
    When the switch is off the original forward is returned untouched.
    """
    options = dict(options or {})
    if not options.get("compile_blocks", False):
        return original

    compile_options = dict(options.get("compile") or {})

    @functools.wraps(original)
    def wrapped(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor = None,
        pos_embeds: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        if position_embeddings is None or cu_seqlens is None or max_seqlen is None:
            _, position_embeddings, cu_seqlens, _, max_seqlen = self.preprcess_grid_thw(grid_thw)
        if pos_embeds is None:
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        hidden_states = hidden_states + pos_embeds
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            _compile_block_forward(blk, compile_options)
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    return wrapped


def forward_without_grid_thw(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor = None,
    pos_embeds: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """Stock forward kept for reference: same body as :func:`compile_vision_blocks`.

    Not referenced by the Catalog; it is the uncompiled counterpart of the
    wrapper above, useful when reading or bisecting the compile path.
    """
    hidden_states = self.patch_embed(hidden_states)

    if position_embeddings is None or cu_seqlens is None or max_seqlen is None:
        _, position_embeddings, cu_seqlens, _, max_seqlen = self.preprcess_grid_thw(grid_thw)
    if pos_embeds is None:
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    hidden_states = hidden_states + pos_embeds
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


class DisabledStepCounter:
    """Monotonic counter whose in-place add stays out of the traced graph.

    Replaces the plain ``int`` so Dynamo has no value to specialise on.  The
    value itself is preserved, in case a downstream consumer is ever added back.
    """

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    @torch._dynamo.disable
    def __iadd__(self, other):
        self.value += int(other)
        return self

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"DisabledStepCounter({self.value})"


def wrap_init(original, options=None):
    """Return an ``__init__`` wrapper that installs :class:`DisabledStepCounter`.

    ``FlowMatchingV2.__init__`` assigns ``self.steps = 0`` (only when
    ``align_params`` is non-empty); the wrapper swaps that ``int`` for the
    Dynamo-safe object right after the constructor returns.
    """

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        steps = self.__dict__.get("steps", None)
        if isinstance(steps, int):
            self.__dict__["steps"] = DisabledStepCounter(steps)
        return None

    return wrapped
