# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""openvla optimization declarations.

Add model-specific Groups here. Importing this module registers the declarations
with TurboPhysAI; the generated OptimizationConfig loads it through optimization_modules.

Declarations only describe target/replacement pairs as strings; nothing here imports
OpenVLA/prismatic and nothing resolves runtime objects at import time.
"""

from __future__ import annotations

from ....engine.definitions import group, replace, wrap

# --- BF16 mixed-precision support detection ---------------------------------
BF16_SUPPORT = group(
    "openvla.bf16_support",
    replace(
        target="prismatic.util.torch_utils.check_bloat16_supported",
        aliases=(
            "prismatic.util.check_bloat16_supported",
            "prismatic.training.strategies.base_strategy.check_bloat16_supported",
        ),
        replacement="turbo_physai.optimizations.models.openvla.fixbf16support.check_bloat16_supported",
    ),
)

# --- FSDP1 per-layer torch.compile -------------------------------------------
COMPILE_FSDP1 = group(
    "openvla.compile.fsdp1",
    wrap(
        target="prismatic.training.strategies.fsdp.FSDPStrategy.run_setup",
        replacement="turbo_physai.optimizations.models.openvla.compile_fsdp1.run_setup_wrapper",
    ),
    wrap(
        target="prismatic.training.strategies.fsdp.FSDPStrategy.save_checkpoint",
        replacement="turbo_physai.optimizations.models.openvla.compile_fsdp1.save_checkpoint_wrapper",
    ),
    wrap(
        target="timm.models.vision_transformer.VisionTransformer._intermediate_layers",
        replacement="turbo_physai.optimizations.models.openvla.vision_timm.timm_intermediate_layers_wrapper",
    ),
)

# --- Fused AdamW (Group enabled => fused is the default) --------------------
FUSED_ADAMW = group(
    "openvla.adamw.fused",
    wrap(
        target="torch.optim.AdamW",
        replacement="turbo_physai.optimizations.models.openvla.fusedAdamW.adamw_fused_wrapper",
    ),
)

# --- FSDP1 communication overlap (limit_all_gathers / fwd+bwd prefetch) ------
FSDP_PREFETCH = group(
    "openvla.fsdp.prefetch",
    wrap(
        target="torch.distributed.fsdp.FullyShardedDataParallel",
        aliases=("prismatic.training.strategies.fsdp.FSDP",),
        replacement="turbo_physai.optimizations.models.openvla.fsdp_prefetch.fsdp_prefetch_wrapper",
    ),
)

# --- Fixed-length (bucketed) text padding ------------------------------------
TEXT_LEN_BUCKET = group(
    "openvla.data.text_len_bucket",
    wrap(
        target="prismatic.util.data_utils.PaddedCollatorForActionPrediction.__call__",
        replacement="turbo_physai.optimizations.models.openvla.text_len_bucket.bucketed_collate_wrapper",
    ),
)

# --- Skip FA2 varlen (unpad) on right-padded prefill -------------------------
SKIP_FA2_UNPAD = group(
    "openvla.llm.skip_fa2_unpad",
    wrap(
        target="transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask",
        replacement="turbo_physai.optimizations.models.openvla.skip_fa2_unpad.make_fast_fa2_causal_mask_wrapper",
    ),
)

# --- bf16 CrossEntropy: drop the fp32 upcast of logits -----------------------
BF16_CE = group(
    "openvla.llm.bf16_ce",
    wrap(
        target="transformers.models.llama.modeling_llama.LlamaForCausalLM.forward",
        replacement="turbo_physai.optimizations.models.openvla.bf16_ce.bf16_ce_forward_wrapper",
    ),
)

# --- RLDS DataLoader: spawned workers (Group options: num_workers, default 1) --
# 仅对 RLDS 数据集（RLDSDataset / EpisodicRLDSDataset）的 DataLoader 强制
# `num_workers=N` + `spawn` context（RLDS 的 TF graph 不能 fork，必须 spawn 重建）；
# RLDS 数据集类包成可 pickle 子类（序列化构造参数、worker 里重建 TF graph）。
DATALOADER_SPAWN = group(
    "openvla.dataloader.spawn",
    wrap(
        target="prismatic.vla.datasets.datasets.RLDSDataset",
        aliases=(
            "prismatic.vla.datasets.RLDSDataset",
            "prismatic.vla.materialize.RLDSDataset",
        ),
        replacement="turbo_physai.optimizations.models.openvla.spawn_dataloader.rlds_dataset_spawn_wrapper",
    ),
    wrap(
        target="prismatic.vla.datasets.datasets.EpisodicRLDSDataset",
        aliases=(
            "prismatic.vla.datasets.EpisodicRLDSDataset",
            "prismatic.vla.materialize.EpisodicRLDSDataset",
        ),
        replacement="turbo_physai.optimizations.models.openvla.spawn_dataloader.episodic_rlds_dataset_spawn_wrapper",
    ),
    wrap(
        target="torch.utils.data.DataLoader",
        aliases=("prismatic.training.strategies.base_strategy.DataLoader",),
        replacement="turbo_physai.optimizations.models.openvla.spawn_dataloader.dataloader_spawn_wrapper",
    ),
)

# --- Python GC freeze before the training loop -------------------------------
GC_FREEZE = group(
    "openvla.gc.freeze",
    wrap(
        target=(
            "prismatic.training.strategies.base_strategy."
            "TrainingStrategy.run_vla_training"
        ),
        replacement="turbo_physai.optimizations.models.openvla.gc_freeze.gc_freeze_wrapper",
    ),
)
