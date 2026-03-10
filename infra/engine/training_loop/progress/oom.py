from __future__ import annotations

import torch


def recover_if_cuda_oom(trainer, *, epoch: int, step: int, error: BaseException) -> bool:
    if not isinstance(error, (torch.OutOfMemoryError, RuntimeError)):
        return False

    if "out of memory" not in str(error).lower():
        return False

    trainer.optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    skip_count = int(getattr(trainer, "_oom_skip_count", 0)) + 1
    trainer._oom_skip_count = skip_count

    if trainer.accelerator.is_main_process:
        if skip_count <= 3 or skip_count % 10 == 0:
            trainer.logger.warning(
                "Skipping batch due to CUDA OOM at epoch={} step={} skip_count={}: {}",
                epoch,
                step,
                skip_count,
                error,
            )
        if skip_count == 3:
            trainer.logger.warning(
                "Repeated CUDA OOM detected. Consider lowering image size/scales, reducing dn_num_group, or setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            )
    return True
