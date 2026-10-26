# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon: moved into the TurboPhysAI optimization package, with one
# deliberate addition -- see "Missing upload_trace" in the module docstring.

"""``lingbotvla.utils.helper.create_profiler`` ported into TurboPhysAI.

The Group ``lingbot-vla-v2.profiling`` installs :func:`create_profiler` in place of the
model repository implementation, so the profiling path can be optimized without
editing the model repository.

:func:`create_profiler` is a copy of
``lingbotvla/utils/helper.py::create_profiler`` at the validated baseline commit:
same activities, same schedule arithmetic, same ``with_modules=True`` and the same
``torch.profiler.tensorboard_trace_handler`` export.  The only differences are the
``logger`` object, which is a standard library logger here so that this module has no
import-time dependency on the model repository (the original uses
``lingbotvla.utils.logging``), and the :func:`_install_upload_trace_stub` call
described below.

Missing ``upload_trace``
------------------------
The model repository calls ``helper.upload_trace(wandb_project, wandb_name,
trace_dir)`` right after ``profiler.stop()`` (``tasks/vla/train_lingbotvla.py:1049``
and ``tasks/vla/train_pi0.py:671``), but that function is **not defined anywhere in
the repository** -- it has no ``def`` in the entire git history, so the call is
dangling and the profiling path raises ``AttributeError`` at
``global_step == profile_end_step``.

Declaring ``lingbotvla.utils.helper.upload_trace`` as its own Replacement is not an
option: the target attribute does not exist, target resolution fails, and the whole
Group would be blocked (``target.unresolved``) -- taking the working
``create_profiler`` replacement down with it.

Instead :func:`create_profiler` installs a no-op ``upload_trace`` on the helper
module when it runs.  That is the right hook: ``create_profiler`` is called exactly
when ``train.enable_profiling`` is true, which is also the only situation in which
``upload_trace`` is ever reached, so the stub is always in place before the call and
is never installed otherwise.  The profiler trace itself is unaffected -- it is
already written to ``trace_dir`` by the ``tensorboard_trace_handler`` scheduled
above, so nothing is lost by skipping the (unknown, unimplemented) upload.

Importing this module requires Torch; the Catalog only carries the string paths, so
``import turbo_physai`` still does not load Torch.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def upload_trace(wandb_project=None, wandb_name=None, trace_dir=None):
    """No-op stand-in for the helper the model repository calls but never defines.

    Signature matches the call sites (``wandb_project``, ``wandb_name``,
    ``trace_dir``) so that a positional call keeps working.  It deliberately does
    nothing beyond logging: the trace was already exported to ``trace_dir`` by
    ``torch.profiler.tensorboard_trace_handler``, and the original behaviour of
    ``upload_trace`` cannot be reconstructed (it never existed in the repository).
    """

    logger.warning(
        "helper.upload_trace is not implemented in the lingbotvla repository; "
        "skipping the upload. The profiler trace is already on disk: %s "
        "(wandb_project=%s, wandb_name=%s)",
        trace_dir,
        wandb_project,
        wandb_name,
    )


def _install_upload_trace_stub() -> bool:
    """Expose :func:`upload_trace` on ``lingbotvla.utils.helper`` if it is missing.

    Returns ``True`` when a stub was installed, ``False`` when the model repository
    already provides a real implementation (which future versions may).
    """

    try:
        import lingbotvla.utils.helper as helper
    except ImportError:  # pragma: no cover - the target environment always has it
        logger.warning(
            "lingbotvla.utils.helper is not importable; upload_trace stub not installed"
        )
        return False

    if hasattr(helper, "upload_trace"):
        return False

    helper.upload_trace = upload_trace
    logger.info(
        "installed a no-op lingbotvla.utils.helper.upload_trace: the model "
        "repository calls it but never defines it"
    )
    return True


def create_profiler(
    start_step: int, end_step: int, trace_dir: str, record_shapes: bool, profile_memory: bool, with_stack: bool
):
    """
    Creates a profiler to record the CPU and CUDA activities. Default export to trace.json.
    Profile steps in [start_step, end_step).

    Args:
        start_step (int): The step to start recording.
        end_step (int): The step to end recording.
        trace_dir (str): The path to save the profiling result.
        record_shapes (bool): Whether to record the shapes of the tensors.
        profile_memory (bool): Whether to profile the memory usage.
        with_stack (bool): Whether to include the stack trace.
    """

    # Called exactly when train.enable_profiling is true, which is the only case in
    # which the model repository reaches helper.upload_trace() -- see the module
    # docstring.  Installing the stub here keeps this module free of import-time
    # side effects on the model repository.
    _install_upload_trace_stub()

    def handler_fn(p):
        torch.profiler.tensorboard_trace_handler(trace_dir)(p)
        logger.info(f"Profiling result saved at {trace_dir}.")

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    warmup = 0 if start_step == 1 else 1
    wait = start_step - warmup - 1
    active = end_step - start_step
    logger.info(f"build profiler schedule - wait: {wait}, warmup: {warmup}, active: {active}.")
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=1,
        ),
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=True,
        with_stack=with_stack,
    )
    return profiler
