# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by Hygon: kernel-option table for the flex-attention calls made by
# ``lingbotvla/models/vla/lingbot_vla/flex_attention.py``.
#
# Wrapped symbol and its TurboPhysAI replacement:
#   lingbotvla.models.vla.lingbot_vla.flex_attention.flex_attention
#       (that is ``torch.nn.attention.flex_attention.flex_attention`` bound in
#        the module namespace)
#       -> this module's flex_attention_kernel_options wrapper
#
# Declared by the Group ``lingbot-vla-v2.flex-kernel-options`` in catalog.py.
"""Kernel-option table for the LingBot-VLA flex-attention calls.

Why the table is replaced
-------------------------
``lingbotvla/models/vla/lingbot_vla/flex_attention.py`` builds every call from
two module constants::

    FLEX_SPARSE_BLOCK_SIZE = 128
    FLEX_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}

With ``head_dim=128`` and the fp32 attention of ``enable_fp32`` training, that
table makes the Triton template request 72 KiB of shared memory per block while
the HCU (DCU) target only provides 64 KiB, so the kernel cannot be compiled and
the first attention call of the run fails.  ``BLOCK_N`` is what costs the shared
memory (the K and V tiles are ``BLOCK_N * head_dim`` each), so the table below
halves it and drops ``num_stages`` to 1.

A TurboPhysAI ``replace`` cannot target a module-level dict (the checker wants a
callable or property target), so the table is injected at the innermost call:
the ``flex_attention`` name that the model module imported from
``torch.nn.attention`` is wrapped, and the wrapper substitutes the
``kernel_options`` argument.  Both model entry points
(``flex_attention_forward`` and ``flex_attention_with_block_mask``) resolve that
name as a module global at call time, so they pick the wrapper up without being
redeclared; the mask builders and the fp32 attention arithmetic stay as they
are.  Only this module's own reference is wrapped, so callers that import
``flex_attention`` from torch directly keep the torch defaults.

Configuration
-------------
``DEFAULT_KERNEL_OPTIONS`` is used when the Group carries no ``kernel_options``
option.  Passing an empty mapping (``kernel_options: {}``) drops the argument
instead, which leaves the tile choice to the torch build's own heuristic.
"""

import functools

#: Table substituted into every call; keys passed by the caller win, so only the
#: tiles the backend is not allowed to pick for itself belong here.
DEFAULT_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_warps": 4,
    "num_stages": 1,
}


def flex_attention_kernel_options(original, options=None):
    """Build the wrapper that substitutes the ``kernel_options`` of a call.

    ``options`` is the option mapping of the Group; ``kernel_options`` inside it
    is the table to apply (empty to keep the backend defaults).
    """
    if options is not None and "kernel_options" in options:
        table = dict(options["kernel_options"] or {})
    else:
        table = dict(DEFAULT_KERNEL_OPTIONS)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if table:
            # The model passes kernel_options as a keyword argument.
            caller = kwargs.get("kernel_options") or {}
            kwargs["kernel_options"] = {**caller, **table}
        else:
            kwargs.pop("kernel_options", None)
        return original(*args, **kwargs)

    return wrapped
