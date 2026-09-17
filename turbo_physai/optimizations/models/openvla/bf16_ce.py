# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import inspect
from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# Anchors of the transformers source this replacement body was derived from.
_ANCHORS = (
    "logits = logits.float()",
    "loss_fct = CrossEntropyLoss()",
)

# HF's `CrossEntropyLoss()` default and OpenVLA's `IGNORE_INDEX`
# (prismatic/util/data_utils.py, prismatic/vla/datasets/datasets.py) agree,
# so the padded label is correctly excluded.
_IGNORE_INDEX = -100


def _verify_baseline_source(original: Callable) -> None:
    """Fail loudly if the live ``forward`` is not the structure we derived from."""

    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:  # pragma: no cover - no source available
        raise RuntimeError(
            "[BF16 CE] cannot read LlamaForCausalLM.forward source for the version "
            f"guard, refusing to install ({exc})"
        ) from exc
    missing = [anchor for anchor in _ANCHORS if anchor not in source]
    if missing:
        raise RuntimeError(
            "[BF16 CE] transformers' LlamaForCausalLM.forward no longer matches the "
            f"validated structure (missing anchors: {missing}); update "
            "turbo_physai/optimizations/models/openvla/bf16_ce.py against the new source before enabling "
            "this Group."
        )


def bf16_ce_forward_wrapper(
    original: Callable, options: Optional[Mapping[str, Any]] = None
) -> Callable:
    """Wrapper factory ``(original, options) -> wrapped forward``.

    Enabling the Group installs the returned callable in place of
    ``transformers.models.llama.modeling_llama.LlamaForCausalLM.forward``.
    ``options`` (Group options) is accepted for framework ``wrap`` compatibility
    and intentionally unused -- the Group has no configurable behaviour.
    """

    del options

    _verify_baseline_source(original)

    # Resolved lazily so importing this module (which TurboPhysAI does during its
    # preparation phase) does not bind transformers internals earlier than needed.
    from transformers.models.llama.modeling_llama import CausalLMOutputWithPast

    @functools.wraps(original)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, "CausalLMOutputWithPast"]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        # [BF16 CE] The baseline's unconditional `logits = logits.float()` is removed:
        # CE consumes autocast's bf16 logits and the softmax kernel accumulates in
        # fp32 internally. The fp16 fallback below is this port's safety net -- it
        # restores exact baseline behaviour for the one dtype where the trick is
        # invalid (see module docstring), and never fires for bf16.
        if logits.dtype == torch.float16:
            logits = logits.float()

        loss = None
        if labels is not None:
            # [BF16 CE] No `logits[..., :-1, :].contiguous()` copy. Right-pad `labels`
            # with the ignore index, then shift: after dropping the first entry it has
            # one label per logits row, and row T-1 is excluded by ignore_index. Same
            # rows and same normaliser as the baseline, minus the ~2 GiB/step copy.
            shift_labels = F.pad(labels, (0, 1), value=_IGNORE_INDEX)[..., 1:]
            # `reshape` (not `view`) so this stays correct even if `logits` ever
            # arrives non-contiguous; it only copies when it actually has to.
            loss_fct = CrossEntropyLoss(ignore_index=_IGNORE_INDEX)
            shift_logits = logits.reshape(-1, self.config.vocab_size)
            shift_labels = shift_labels.reshape(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward
