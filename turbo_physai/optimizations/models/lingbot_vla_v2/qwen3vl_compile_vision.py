# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen3-VL vision 塔的逐层 ``torch.compile``。

两块内容：

* ``compile_vision_blocks``：包在 ``qwen3vl_in_vla.forward_without_grid_thw`` 外层，
  循环里把每个 block 的 ``forward`` 换成编译版本。
* ``DisabledStepCounter`` / ``wrap_init``：``FlowMatchingV2`` 的 ``self.steps`` 是
  一个普通 ``int``，Dynamo 会按值 guard，每步重编到第 8 步就退回 eager，
  编译过的 block 也就走不到了。

为什么编译 ``blk.forward`` 而不是 ``blk``
----------------------------------------
``GradientCheckpointingLayer.__call__`` 把 checkpoint 包在 block 计算外层。
编译整个 block 对象会让 ``torch.compile`` 落在 checkpoint 外面，Dynamo 碰到
checkpoint HOP 直接 break，重算那一步仍是 eager。编译 ``blk.forward`` 并把它
挂到实例上，``__call__`` 的 checkpoint 分支、FSDP hook、kwargs 处理都保留。
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch

COMPILE_OPTIONS = {"dynamic": True}

#: ``TP_APPLY_LOG=1`` 时打印安装回执，用来区分"装上了"和"根本没加载"。
_APPLY_LOG = os.environ.get("TP_APPLY_LOG", "") == "1"
_apply_logged = [False]


def _drop_view(tensor: torch.Tensor) -> torch.Tensor:
    """输入是 view 时物化一份。

    第一个 block 的输入来自 ``hidden_states.reshape(seq_len, -1)``，在 contiguous
    张量上返回 view（``_base`` 非空）；后续 block 拿到的是新建张量（``_base`` 为空）。
    Dynamo 按 ``_base.size()`` 建 guard，重算时拿到新张量会抛
    ``'NoneType' object has no attribute 'size'``。

    ``contiguous()`` 治不了（已 contiguous 的 view 原样返回），只有拷贝能去掉
    ``_base``。只动布局，不动 dtype 和数值。
    """
    if getattr(tensor, "_base", None) is not None:
        tensor = tensor.clone()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _compile_block_forward(blk):
    """编译一个 block 的 ``forward``，缓存在实例上。

    Dynamo trace 期间直接返回原函数：``torch.compile`` 此时返回入参本身，
    缓存下来会让这个 block 永远停在 eager。
    """
    if torch.compiler.is_compiling():
        return blk.forward
    cached = blk.__dict__.get("_tp_compiled_forward")
    if cached is None:
        cached = torch.compile(blk.forward, **COMPILE_OPTIONS)
        blk.__dict__["_tp_compiled_forward"] = cached
        blk.__dict__["forward"] = cached
    return cached


def compile_vision_blocks(original, _options=None):
    """Group 入口：包住 vision 塔的 ``forward``，循环里逐 block 装编译。"""

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
            _compile_block_forward(blk)
            hidden_states = blk(
                _drop_view(hidden_states),
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
        if _APPLY_LOG and not _apply_logged[0]:
            # 必须在循环之后统计：循环前统计永远是 0/24。
            _apply_logged[0] = True
            installed = sum(
                1 for blk in self.blocks if "_tp_compiled_forward" in blk.__dict__
            )
            print(
                f"[tp-apply] vision tower: {installed}/{len(self.blocks)} blocks wrapped",
                flush=True,
            )
        return hidden_states, deepstack_feature_lists

    return wrapped


class DisabledStepCounter:
    """自增不进图的计数器，替代 ``int`` 让 Dynamo 没有可特化的值。"""

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


def wrap_init(original, _options=None):
    """Group 入口：``FlowMatchingV2.__init__`` 返回后替换 ``self.steps``。"""

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        steps = self.__dict__.get("steps", None)
        if isinstance(steps, int):
            self.__dict__["steps"] = DisabledStepCounter(steps)
        return None

    return wrapped
