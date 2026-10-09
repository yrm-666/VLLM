"""Optional Hugging Face Molmo2 inference adapter for preselected frames."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    input_tokens: int
    visual_tokens: int
    processor_seconds: float
    ttft_seconds: float | None
    generation_seconds: float
    peak_vram_gb: float | None
    output_tokens: int | None = None


class Molmo2HFRunner:
    """Run Molmo2 on an already selected chronological frame array.

    Passing ``do_sample_frames=False`` is essential: it prevents the official
    video processor from uniformly resampling our selected frames a second time.
    Original frame indices remain in ``video_metadata`` so Molmo2 sees correct
    timestamps rather than treating the selected sequence as a dense clip.
    """

    def __init__(
        self,
        model_id: str = "allenai/Molmo2-4B",
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attention_backend: str = "sdpa",
        local_files_only: bool = False,
        revision: str = "main",
    ) -> None:
        try:
            import numpy as np
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Molmo2HFRunner requires numpy, torch, and transformers. See "
                "docs/GPU_RUNBOOK.md."
            ) from error

        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        self._np = np
        self._torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=local_files_only,
            revision=revision,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attention_backend,
            local_files_only=local_files_only,
            revision=revision,
        )
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def _synchronize(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def release_unused_memory(self) -> None:
        """Release allocator leftovers after an unsuccessful example."""
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def generate(
        self,
        query: str,
        frames: Sequence[Any],
        video_metadata: dict[str, Any],
        *,
        max_new_tokens: int = 128,
        measure_ttft: bool = False,
    ) -> GenerationResult:
        if len(frames) == 0:
            raise ValueError("at least one selected frame is required")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        frame_array = self._np.stack(list(frames), axis=0)
        metadata = dict(video_metadata)
        metadata["frames_indices"] = self._np.asarray(
            metadata["frames_indices"], dtype=self._np.float64
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "video", "video": "predecoded-selected-frames"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        processor_start = perf_counter()
        inputs = self.processor(
            text=text,
            videos=[frame_array],
            video_metadata=[metadata],
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )
        processor_seconds = perf_counter() - processor_start
        inputs = self._move_inputs(dict(inputs))
        input_tokens = int(inputs["input_ids"].shape[-1])
        grid = inputs.get("video_grids")
        visual_tokens = int(grid[0].prod().item()) if grid is not None else 0

        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()

        ttft_seconds: float | None = None
        if measure_ttft:
            self._synchronize()
            ttft_start = perf_counter()
            with self._torch.inference_mode():
                self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
            self._synchronize()
            ttft_seconds = perf_counter() - ttft_start

        self._synchronize()
        generation_start = perf_counter()
        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        self._synchronize()
        generation_seconds = perf_counter() - generation_start

        generated_tokens = generated_ids[:, input_tokens:]
        text_output = self.processor.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        peak_vram_gb = None
        if self._torch.cuda.is_available():
            peak_vram_gb = self._torch.cuda.max_memory_allocated() / (1024**3)

        return GenerationResult(
            text=text_output,
            input_tokens=input_tokens,
            visual_tokens=visual_tokens,
            processor_seconds=processor_seconds,
            ttft_seconds=ttft_seconds,
            generation_seconds=generation_seconds,
            peak_vram_gb=peak_vram_gb,
            output_tokens=int(generated_tokens.shape[-1]),
        )
