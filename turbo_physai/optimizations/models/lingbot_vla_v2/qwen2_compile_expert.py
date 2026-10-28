# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen2 action expert（MoE decoder）的逐层 ``torch.compile``。

和 text 塔同一套做法：把 ``layer.forward`` 挂到实例上，这样
``GradientCheckpointingLayer.__call__`` 的 checkpoint 仍然包着它。

只编 ``compute_kqv``
--------------------
每层每步被调用两次，各自一个 checkpoint 作用域：

* ``compute_kqv=True`` -> Q/K/V 投影，纯 ATen，一个图就够。**编译**。
* ``output_atten=True`` -> o_proj + 残差 + MoE。**留 eager**：
  ``FusedMoeExpertFunction.apply`` / ``moe_scatter`` / ``group_gemm`` 是 Triton
  自定义 autograd.Function，Dynamo trace 不了。

所以每层一张图，不是两张。

安装点是 ``Qwen2ForCausalLM.__init__``（不是共享的
``QwenvlWithExpertV2Model.__init__``），这样和 text 塔的 Group 互不影响。
"""

from __future__ import annotations

import functools
import os
import types
from typing import Any, Optional

import torch

COMPILE_OPTIONS = {"dynamic": True}

#: ``TP_APPLY_LOG=1`` 时打印安装回执，用来区分"装上了"和"根本没加载"。
_APPLY_LOG = os.environ.get("TP_APPLY_LOG", "") == "1"


def _kqv_impl(self, hidden_states, ada_cond=None):
    """``Qwen2DecoderLayer.forward`` 的 ``compute_kqv`` 分支。

    两个 ``.to(param_dtype)`` 是 ``forward`` 原有的，不是新加的；进这里时
    ``hidden_states`` 已经是 ``param_dtype``，第一个是 no-op。
    """
    param_dtype = self.self_attn.q_proj.weight.dtype
    hidden_states = hidden_states.to(param_dtype)
    if ada_cond is not None:
        ada_cond = ada_cond.to(param_dtype)

    if ada_cond is not None:
        hidden_states = self.input_layernorm(hidden_states, ada_cond)
    else:
        hidden_states = self.input_layernorm(hidden_states)
    hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

    query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
    key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
    value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
    return query_state, key_state, value_state


def _stabilise(tensor, param_dtype):
    """转成权重 dtype 并 copy 一份，在 checkpoint 之前调用。

    ``Qwen2DecoderLayer.forward`` 开头的 ``.to(param_dtype)`` 在 checkpoint 内部
    执行，于是 checkpoint 存下来的是转换前的 fp32；重算时 autograd 记的却是转换
    后的 bf16。同一个张量两种 dtype，MoE 的 ``group_gemm`` 会走不同路由，最终

        CheckpointError: forward: 16, recomputation: 9
        forward  saved[0] = [32, 51, 768] float32
        recompute saved[1] = [32, 51, 768] bfloat16

    这里把官方那行 ``.to(param_dtype)`` 提前到 checkpoint 捕获参数之前，两趟
    喂给 body 的张量完全一致。数值和 dtype 与 eager 相同，``clone`` 只改布局。
    """
    if tensor is None:
        return None
    return tensor.to(param_dtype).clone()


def _compile_layer_forward(layer) -> Any:
    """给一层编 ``compute_kqv`` 一张图，把 ``hybrid`` 挂到实例 ``forward`` 上。"""
    if torch.compiler.is_compiling():
        return layer.forward
    hybrid = layer.__dict__.get("_tp_compiled_forward")
    if hybrid is None:
        eager_forward = layer.forward
        bound_kqv = types.MethodType(_kqv_impl, layer)
        compiled_kqv = torch.compile(bound_kqv, **COMPILE_OPTIONS)
        param_dtype = layer.self_attn.q_proj.weight.dtype

        # 存在实例 __dict__ 里，Python 不会自动绑定 self；参数就是调用方传的。
        def hybrid(*args, **kwargs):
            if kwargs.get("output_atten", False) or (len(args) > 5 and args[5]):
                return eager_forward(*args, **kwargs)
            hidden_states = args[0] if args else kwargs["hidden_states"]
            if hidden_states.dtype is not param_dtype:
                # 不转换，走 eager，让图里永远只有一种 dtype 可 guard。
                return eager_forward(*args, **kwargs)
            if not hidden_states.is_contiguous():
                hidden_states = hidden_states.contiguous()
            ada_cond = kwargs.get("ada_cond")
            if ada_cond is None and len(args) > 6:
                ada_cond = args[6]
            return compiled_kqv(hidden_states, ada_cond=ada_cond)

        layer.__dict__["_tp_compiled_forward"] = hybrid
        layer.__dict__["forward"] = hybrid
    return hybrid


def _install_call_guard() -> None:
    """在 checkpoint 捕获参数之前，把 expert 层的输入统一成 ``param_dtype``。

    ``GradientCheckpointingLayer.__call__`` 存的是它收到的 ``*args``，所以
    ``forward`` 内部做的任何 dtype 转换对账本都不可见 —— 这正是 ``16 vs 9`` 的
    成因（详见 :func:`_stabilise`）。

    ``__call__`` 是模型调用链上最后一个还在 checkpoint 外面的位置，而且只能打在
    **类**上：``obj(...)`` 解析的是 ``type(obj).__call__``，写进实例 ``__dict__``
    永远不会被调用。
    """
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        Qwen2DecoderLayer,
    )

    if Qwen2DecoderLayer.__dict__.get("_tp_call_guard", False):
        return
    original_call = Qwen2DecoderLayer.__call__

    def guarded_call(self, *args, **kwargs):
        param_dtype = self.self_attn.q_proj.weight.dtype
        if args:
            args = list(args)
            for index in (0, 1):
                if index < len(args) and torch.is_tensor(args[index]):
                    args[index] = _stabilise(args[index], param_dtype)
            args = tuple(args)
        ada_cond = kwargs.get("ada_cond")
        if torch.is_tensor(ada_cond):
            kwargs["ada_cond"] = _stabilise(ada_cond, param_dtype)
        return original_call(self, *args, **kwargs)

    Qwen2DecoderLayer.__call__ = guarded_call
    Qwen2DecoderLayer._tp_call_guard = True


def _expert_layers(model) -> Optional[list]:
    """返回 action expert 的 decoder 层，取不到返回 ``None``。"""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return None
    return list(layers)


def _ensure_compiled(model) -> None:
    """每个模型实例只装一次（编译 + ``__call__`` 守卫）。"""
    if torch.compiler.is_compiling():
        return
    _install_call_guard()
    if model.__dict__.get("_tp_expert_layers_compiled", False):
        return
    layers = _expert_layers(model)
    if layers is None:
        return
    for layer in layers:
        _compile_layer_forward(layer)
    model.__dict__["_tp_expert_layers_compiled"] = True
    if _APPLY_LOG:
        installed = sum(
            1 for layer in layers if "_tp_compiled_forward" in layer.__dict__
        )
        print(
            f"[tp-apply] expert tower: {installed}/{len(layers)} layers wrapped "
            f"call_guard=True",
            flush=True,
        )


def install_on_init(original, _options=None):
    """Group 入口：``Qwen2ForCausalLM`` 构造完成后装编译和 ``__call__`` 守卫。"""

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _ensure_compiled(self)
        return None

    return wrapped
