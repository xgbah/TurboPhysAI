# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by Hygon: compile the frozen distillation teachers' forward passes.
#
# Wrapped symbols and their TurboPhysAI replacements:
#   lingbotvla.models.vla.vision_models.module_utils.v2.forward
#       (MoGeModel.forward, reached by MoGeModel.infer -> self.forward)
#   lingbotvla.models.vla.vision_models.module_utils.v2_morgbd.forward_feat
#       (MDMModel.forward_feat, reached by MDMModel.infer_feat -> self.forward_feat)
#   lingbotvla.models.vla.vision_models.dino_video.lumos_dinov3.models.inference
#       .CausalEvalAdapter.get_intermediate_layers
#       (reached by DinoVideoTeacher.get_future_feature -> self.adapter...)
#       -> this module's compiled_forward wrapper
#
# Declared by the Group ``lingbot-vla-v2.compile`` in catalog.py.
"""Compile the frozen teacher forwards that build the distillation targets.

Why the teachers ran eager
--------------------------
``train_lingbotvla.py`` compiles the teachers with ``torch.compile(moge_model)``,
which only patches ``OptimizedModule.__call__``.  The training loop calls
``moge_model.infer(...)`` / ``morgbd_model.infer_feat(...)`` instead; those are
custom methods forwarded to ``_orig_mod``, and they reach ``self.forward`` /
``self.forward_feat`` -- the original, uncompiled functions.  The video teacher
has the same shape one level down: ``get_future_feature`` calls
``self.adapter.get_intermediate_layers``.  Every depth/video target was
therefore produced one ATen kernel at a time (the profile shows ~1200-1800
kernels per call and no Inductor fusion).

What this wrapper compiles
--------------------------
Only the three compute entry points, i.e. the work ``infer`` /
``get_future_feature`` perform inside their ``autocast`` / ``inference_mode`` /
``no_grad`` scopes.  The host-side prologue and epilogue (dtype casts, shape
checks, ``squeeze``, ``nan_to_num``, dict assembly) stays eager, which costs
nothing and keeps the compiled graph free of Python control flow.
``torch.compile`` is applied lazily on the first call so that import and Group
preparation stay side-effect free, and the compiled callable is cached for
every later step.

The video adapter already compiles its FlexAttention call site by itself
(``flex_attention_packed._flex_attention_compiled``), so Dynamo sees a compiled
callable inside and breaks the graph there.  That boundary is expected: the
attention kernel keeps its own tuning and everything around it is still fused.

Configuration
-------------
``options["compile"]`` is passed verbatim to ``torch.compile``; for a fixed
input shape this is e.g. ``{"mode": "max-autotune", "dynamic": False}``.  An
empty mapping keeps the ``torch.compile`` defaults.
"""

from __future__ import annotations

import functools

import torch


def compiled_forward(original, options=None):
    """Return a wrapper that compiles ``original`` on its first call."""

    compile_options = dict((options or {}).get("compile") or {})
    compiled = None

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(original, **compile_options)
        return compiled(*args, **kwargs)

    return wrapped
