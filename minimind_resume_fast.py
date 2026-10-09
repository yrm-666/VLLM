"""Resume the official MiniMind pretrainer after changing its microbatch size.

The original checkpoint is backed up before training starts.
Only the checkpoint's batch index is converted; model and optimizer states are
loaded by the original trainer. The original source files are not edited.
"""

import argparse
import os
from pathlib import Path
import runpy
import shutil
import sys
from datetime import datetime


def converted_step(step, old_batch, old_accumulation, new_batch, new_accumulation):
    if min(old_batch, old_accumulation, new_batch, new_accumulation) < 1:
        raise ValueError("Batch sizes and accumulation steps must be positive.")
    if old_batch * old_accumulation != new_batch * new_accumulation:
        raise ValueError("Keep batch_size * accumulation_steps unchanged when resuming.")
    # Gradients between optimizer updates are not stored by the official trainer.
    # Replay that small unfinished group instead of silently losing those samples.
    committed_step = step - step % old_accumulation
    return committed_step * old_batch // new_batch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-dir", default=r"C:\Users\hjy\minimind")
    parser.add_argument("--previous-batch-size", type=int, required=True)
    parser.add_argument("--previous-accumulation-steps", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    wrapper, trainer_argv = parser.parse_known_args()

    settings = argparse.ArgumentParser(add_help=False)
    settings.add_argument("--batch_size", type=int, default=32)
    settings.add_argument("--accumulation_steps", type=int, default=8)
    settings.add_argument("--hidden_size", type=int, default=768)
    settings.add_argument("--use_moe", type=int, default=0)
    settings.add_argument("--save_weight", default="pretrain")
    settings.add_argument("--from_resume", type=int, default=0)
    new, _ = settings.parse_known_args(trainer_argv)
    if new.from_resume != 1:
        raise ValueError("This launcher requires --from_resume 1.")

    project = Path(wrapper.project_dir).resolve()
    script = project / "trainer" / "train_pretrain.py"
    suffix = "_moe" if new.use_moe else ""
    checkpoint = project / "checkpoints" / f"{new.save_weight}_{new.hidden_size}{suffix}_resume.pth"
    if not script.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Trainer or checkpoint missing: {script}, {checkpoint}")

    os.chdir(project / "trainer")
    sys.path.insert(0, str(project))
    import datasets  # noqa: F401; match the official Windows DLL import order.
    from trainer import trainer_utils

    original_checkpoint = trainer_utils.lm_checkpoint

    def resume_checkpoint(config, *positional, **kwargs):
        if kwargs.get("model") is not None:
            kwargs["resume_batch_size"] = new.batch_size
            kwargs["resume_accumulation_steps"] = new.accumulation_steps
            return original_checkpoint(config, *positional, **kwargs)

        data = original_checkpoint(config, *positional, **kwargs)
        if data is None:
            raise FileNotFoundError("No resumable checkpoint found.")
        old_batch = data.get("resume_batch_size", wrapper.previous_batch_size)
        old_accumulation = data.get("resume_accumulation_steps", wrapper.previous_accumulation_steps)
        old_step = int(data["step"])
        data["step"] = converted_step(
            old_step, old_batch, old_accumulation,
            new.batch_size, new.accumulation_steps,
        )
        print(
            f"Resume: old batch={old_batch}, step={old_step}; "
            f"new batch={new.batch_size}, step={data['step']}; "
            f"effective batch={new.batch_size * new.accumulation_steps}.",
            flush=True,
        )
        return data

    if wrapper.dry_run:
        from types import SimpleNamespace
        data = resume_checkpoint(
            SimpleNamespace(hidden_size=new.hidden_size, use_moe=bool(new.use_moe)),
            weight=new.save_weight,
            save_dir="../checkpoints",
        )
        print(f"Dry run only. Epoch index: {data['epoch']}. No training or checkpoint writes.")
        return

    backup_dir = Path(__file__).resolve().parent / "minimind_checkpoint_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{checkpoint.stem}_{datetime.now():%Y%m%d_%H%M%S}.pth"
    shutil.copy2(checkpoint, backup)
    print(f"Original checkpoint backup: {backup}", flush=True)
    trainer_utils.lm_checkpoint = resume_checkpoint
    sys.argv = [str(script), *trainer_argv]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
