# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""lingbot_vla_v2 optimization declarations.

Add model-specific Groups here. Importing this module registers the declarations
with TurboPhysAI; the generated OptimizationConfig loads it through optimization_modules.

Declarations only describe target/replacement pairs as strings; nothing here imports
lingbotvla and nothing resolves runtime objects at import time.

The package directory name must stay a valid Python identifier (``lingbot_vla_v2``),
because ``optimization_modules`` and every ``replacement`` path is imported through
``importlib``.  ``--model`` also normalizes ``-`` to ``_``, so a directory named
``lingbot_vla_v2`` is reachable as both ``--model lingbot-vla-v2`` and
``--model lingbot_vla_v2``, while a hyphenated directory name would be unreachable.
"""

from __future__ import annotations

from ....engine.definitions import group, replace, wrap

# --- Profiling: torch profiler setup -----------------------------------------
PROFILER = group(
    "lingbot-vla-v2.profiling",
    replace(
        target="lingbotvla.utils.helper.create_profiler",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.profiling.create_profiler"
        ),
    ),
)

# --- Muon optimizer: DTensor-aware Newton-Schulz orthogonalization -------------
MUON = group(
    "lingbot-vla-v2.muon",
    replace(
        target="lingbotvla.optim.muon.DistributedMuon",
        aliases=("lingbotvla.optim.optimizer.DistributedMuon",),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.muon.DistributedMuon"
        ),
    ),
    replace(
        target="lingbotvla.optim.muon.split_muon_adamw_params",
        aliases=("lingbotvla.optim.optimizer.split_muon_adamw_params",),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.muon.split_muon_adamw_params"
        ),
    ),
    replace(
        target="lingbotvla.optim.muon.batched_newton_schulz",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.muon.batched_newton_schulz"
        ),
    ),
)

# --- Attention: tuned flex-attention kernel-option table ----------------------
FLEX_KERNEL_OPTIONS = group(
    "lingbot-vla-v2.flex-kernel-options",
    wrap(
        target="lingbotvla.models.vla.lingbot_vla.flex_attention.flex_attention",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.flex_kernel_options."
            "flex_attention_kernel_options"
        ),
    ),
)

# --- Vision tower: patch embedding as a GEMM instead of a Conv3d --------------
VISION_PATCH_EMBED = group(
    "lingbot-vla-v2.conv3d-to-linear",
    replace(
        target=(
            "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionPatchEmbed."
            "forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.conv3d_to_linear."
            "forward"
        ),
    ),
)

# --- Vision tower: cache the position-embedding index/weight tables ----------
QWEN3VL_CACHE_GRID_THW = group(
    "lingbot-vla-v2.qwen3vl-cache-grid-thw",
    replace(
        target=(
            "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla."
            "Qwen3VLVisionModel.fast_pos_embed_interpolate"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_cache_gridTHW."
            "fast_pos_embed_interpolate"
        ),
    ),
)

# --- Qwen3-VL: make torch.compile effective (steps guard + block compile) ----
# Both entries are needed for the compiled path to stay compiled: the dead
# ``self.steps`` counter must not re-specialise every step, and the vision
# blocks are compiled inside the checkpoint wrapper.
# ``wrap`` (not ``replace``) so the Group's options reach the implementation:
# ``compile_blocks`` switches on the per-layer ``torch.compile`` path and
# ``compile`` is forwarded verbatim to ``torch.compile``.
QWEN3VL_COMPILE = group(
    "lingbot-vla-v2.qwen3vl-compile",
    wrap(
        target=(
            "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2."
            "FlowMatchingV2.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_compile_vision."
            "wrap_init"
        ),
    ),
    wrap(
        target=(
            "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla."
            "forward_without_grid_thw"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_compile_vision."
            "compile_vision_blocks"
        ),
    ),
)

# --- Activation checkpoint: keep Dynamo's tracing out of the bookkeeping ------
# The stock non-reentrant checkpoint keeps a positional list of the tensors
# autograd saves (``frame.weak_holders``).  Dynamo traces a compiled block
# *inside* that scope: the tracing pass saves a few tensors of its own, they are
# appended ahead of the real forward's, and ``check_recomputed_tensors_match``
# then compares a shifted list (``position 0: [32,768] vs [768,4096]``).
# The patch makes ``pack_hook`` register nothing while a trace is running, so
# the list only ever contains what the real compiled execution saved.
CHECKPOINT_TRACE_GUARD = group(
    "lingbot-vla-v2.checkpoint-trace-guard",
    wrap(
        target="torch.utils.checkpoint._checkpoint_hook",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2."
            "checkpoint_trace_guard.install"
        ),
    ),
)

# --- Data parallel: FSDP1 compatibility shims ---------------------------------
# Selecting FSDP1 is `--train.data_parallel_mode fsdp1`; the branch already
# exists.  These shims fix the two places where that branch assumes the old
# defaults: FSDP2's "no reshard after forward" (``enable_full_shard: false``)
# must become ``SHARD_GRAD_OP`` instead of FSDP1's ``NO_SHARD``, and
# ``MixedPrecision(buffer_dtype=fp32)`` must drop the fp32 buffer dtype or
# Dynamo's fake-tensor tracing trips over ``set_(fp32, bf16)``.
FSDP1_COMPAT = group(
    "lingbot-vla-v2.fsdp1-compat",
    wrap(
        target=(
            "lingbotvla.distributed.torch_parallelize."
            "build_parallelize_model"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2."
            "use_fadp1.install"
        ),
    ),
)

# --- Teachers: compile the frozen distillation forwards ----------------------
TEACHER_COMPILE = group(
    "lingbot-vla-v2.compile-teacher",
    wrap(
        target="lingbotvla.models.vla.vision_models.module_utils.v2.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2."
            "compile_teacher.compiled_forward"
        ),
    ),
    wrap(
        target=(
            "lingbotvla.models.vla.vision_models.module_utils."
            "v2_morgbd.forward_feat"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2."
            "compile_teacher.compiled_forward"
        ),
    ),
    # Video / DINOv3 teacher: ``DinoVideoTeacher.get_future_feature`` ->
    # ``self.adapter.get_intermediate_layers``.  Without this entry the whole
    # DINOv3 backbone runs one ATen kernel at a time (profiled as
    # ``layers/block.py:309 forward`` in python_function, never inside a
    # ``Torch-Compiled Region``).
    wrap(
        target=(
            "lingbotvla.models.vla.vision_models.dino_video.lumos_dinov3."
            "models.inference.CausalEvalAdapter.get_intermediate_layers"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2."
            "compile_teacher.compiled_forward"
        ),
    ),
)
