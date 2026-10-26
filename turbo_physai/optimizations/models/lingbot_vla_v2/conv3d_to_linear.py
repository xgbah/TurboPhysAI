# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn.functional as F


def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Upstream ``Qwen3VLVisionPatchEmbed.forward`` as a matrix product."""

    weight = self.proj.weight
    hidden_states = hidden_states.to(dtype=weight.dtype)
    hidden_states = hidden_states.view(
        -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
    )

    hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)
    weight = weight.reshape(weight.shape[0], -1)
    return F.linear(hidden_states, weight, self.proj.bias)
