# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""逐层 ``torch.compile`` Qwen3-VL text decoder。

每层每步被调用两次，各自是一个 checkpoint 作用域：

* ``compute_kqv=True``  -> ``_kqv_impl``：input_layernorm + q/k/v_proj + q/k_norm。
* ``output_atten=True`` -> ``_oa_impl``：o_proj + 两次残差 + post_layernorm + 稠密 MLP。

两半分别编译。合成一个图会让 Dynamo 同时 guard ``att_output``（Q/K/V 调用时是
``None``、attention 时是张量）和 dtype，图被反复淘汰重编。

安装点是 ``QwenvlWithExpertV2Model.__init__``：``torch.compile`` 在
``is_compiling()`` 为真时原样返回入参，从被 trace 的 ``forward`` 里安装会把
eager 函数当成"已编译"缓存下来。

只处理 ``self.qwenvl.model.language_model.layers``；action expert 有独立的 Group。
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


def _kqv_impl(self, hidden_states):
    """``Qwen3VLTextDecoderLayer.forward`` 的 ``compute_kqv`` 分支。

    用 ``types.MethodType`` 绑定：Dynamo 按类型 guard ``self``，闭包捕获
    ``layer`` 会按身份 guard，一层一张图。
    """
    param_dtype = self.self_attn.q_proj.weight.dtype
    hidden_states = hidden_states.to(param_dtype)

    hidden_states = self.input_layernorm(hidden_states)
    hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

    query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
    key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
    value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
    return query_state, key_state, value_state


def _oa_impl(self, hidden_states, att_output, start, end):
    """``Qwen3VLTextDecoderLayer.forward`` 的 ``output_atten`` 分支。

    elementwise 最密的一半：o_proj + 两次残差 + RMSNorm + MLP 的 SiLU/mul，
    编译后 Inductor 能把它们融进同一批 pointwise kernel。

    这里的 ``.to(param_dtype)`` 是 ``forward`` 原有的前置转换（177-179 行）和
    ``o_proj`` 守卫（192-193 行），没有新增 cast。
    """
    param_dtype = self.self_attn.q_proj.weight.dtype
    hidden_states = hidden_states.to(param_dtype)
    if att_output is not None:
        att_output = att_output.to(param_dtype)

    if att_output.dtype != self.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
    out_emb = self.self_attn.o_proj(att_output[:, start:end])

    # first residual
    out_emb += hidden_states
    after_first_residual = out_emb.clone()
    out_emb = self.post_attention_layernorm(out_emb)
    out_emb = self.mlp(out_emb)
    # second residual
    out_emb += after_first_residual

    return out_emb


def _stabilise(tensor, param_dtype):
    """转成权重 dtype 并物化，让 forward 和 recompute 走同一张图。

    ``.to(param_dtype)`` 是 ``forward`` 原有的前置转换，挪到图外。``clone`` 针对
    布局：forward 时输入是 view（``_base`` 非空），recompute 时是新张量（``_base``
    为空），Dynamo 按 ``_base.size()`` 建的 guard 会炸；``contiguous()`` 治不了
    （已 contiguous 的 view 原样返回），只有拷贝能去掉 ``_base``。

    无条件拷贝是刻意的：只在 ``_base`` 非空时拷贝会让两趟走不同算子序列。
    """
    tensor = tensor.to(param_dtype)
    tensor = tensor.clone()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _compile_layer_forward(layer) -> Any:
    """给一层的两半各编一张图，把 ``hybrid`` 挂到实例 ``forward`` 上。

    ``GradientCheckpointingLayer.__call__`` 的 checkpoint 在层计算之外，
    被替换的 ``forward`` 是首次执行和重算都会走到的入口。

    路由只看调用形态，不看 dtype：``output_atten`` 不做 dtype 闸，否则 forward
    和 recompute 可能走不同路径。其他形态一律回落到原始 ``forward``。
    """
    if torch.compiler.is_compiling():
        return layer.forward
    hybrid = layer.__dict__.get("_tp_compiled_forward")
    if hybrid is None:
        eager_forward = layer.forward
        compiled_kqv = torch.compile(
            types.MethodType(_kqv_impl, layer), **COMPILE_OPTIONS
        )
        compiled_oa = torch.compile(
            types.MethodType(_oa_impl, layer), **COMPILE_OPTIONS
        )
        param_dtype = layer.self_attn.q_proj.weight.dtype

        # 存在实例 __dict__ 里，Python 不会自动绑定 self；参数就是调用方传的。
        def hybrid(*args, **kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]

            compute_kqv = kwargs.get("compute_kqv", False)
            if not compute_kqv and len(args) > 4:
                compute_kqv = args[4]
            if compute_kqv:
                att_output = args[1] if len(args) > 1 else kwargs.get("att_output")
                if att_output is None:
                    return compiled_kqv(_stabilise(hidden_states, param_dtype))
                return eager_forward(*args, **kwargs)

            output_atten = kwargs.get("output_atten", False)
            if not output_atten and len(args) > 5:
                output_atten = args[5]
            if output_atten:
                att_output = args[1] if len(args) > 1 else kwargs.get("att_output")
                if att_output is not None:
                    start = args[2] if len(args) > 2 else kwargs.get("start", 0)
                    end = args[3] if len(args) > 3 else kwargs.get("end", 0)
                    return compiled_oa(
                        _stabilise(hidden_states, param_dtype),
                        _stabilise(att_output, param_dtype),
                        start,
                        end,
                    )

            return eager_forward(*args, **kwargs)

        layer.__dict__["_tp_compiled_forward"] = hybrid
        layer.__dict__["forward"] = hybrid
    return hybrid


def _text_layers(model) -> Optional[list]:
    """返回 Qwen3-VL text decoder 的层，取不到返回 ``None``。"""
    qwenvl = getattr(model, "qwenvl", None)
    inner = getattr(qwenvl, "model", None)
    language_model = getattr(inner, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return None
    return list(layers)


def _ensure_compiled(model) -> None:
    """每个模型实例只装一次。

    Dynamo trace 期间直接返回：``torch.compile`` 此时原样返回入参，缓存下来会
    让这些层永远停在 eager。``__init__`` 包装器是真正的安装点，这里只是首次
    eager 执行的兜底。
    """
    if torch.compiler.is_compiling():
        return
    if model.__dict__.get("_tp_text_layers_compiled", False):
        return
    layers = _text_layers(model)
    if layers is None:
        return
    for layer in layers:
        _compile_layer_forward(layer)
    model.__dict__["_tp_text_layers_compiled"] = True
    if _APPLY_LOG:
        installed = sum(
            1 for layer in layers if "_tp_compiled_forward" in layer.__dict__
        )
        print(
            f"[tp-apply] text tower: {installed}/{len(layers)} layers wrapped",
            flush=True,
        )


def install_on_init(original, _options=None):
    """Group 入口：构造完成后给 text 层装上编译。"""

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _ensure_compiled(self)
        return None

    return wrapped


def compile_text_layers(original, _options=None):
    """Group 入口：``forward`` 首次 eager 执行时的兜底安装。"""

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        _ensure_compiled(self)
        return original(self, *args, **kwargs)

    return wrapped
