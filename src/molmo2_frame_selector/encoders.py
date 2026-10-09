"""Dual-encoder interfaces and the optional SigLIP2 implementation."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DualEncoder(Protocol):
    """Minimal contract required by the selection pipeline."""

    @property
    def fingerprint(self) -> str: ...

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_images(self, images: Sequence[Any]) -> list[list[float]]: ...


def l2_normalize(vector: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    values = [float(value) for value in vector]
    if not values:
        raise ValueError("embedding cannot be empty")
    if any(not isfinite(value) for value in values):
        raise ValueError("embedding must contain only finite values")
    norm = sqrt(sum(value * value for value in values))
    if norm <= epsilon:
        raise ValueError("embedding norm is zero")
    return [value / norm for value in values]


def cosine_similarities(
    query_embedding: Sequence[float],
    frame_embeddings: Sequence[Sequence[float]],
) -> list[float]:
    query = l2_normalize(query_embedding)
    output: list[float] = []
    for frame_embedding in frame_embeddings:
        frame = l2_normalize(frame_embedding)
        if len(frame) != len(query):
            raise ValueError("text and image embeddings must have equal dimensions")
        similarity = sum(left * right for left, right in zip(query, frame))
        output.append(max(-1.0, min(1.0, similarity)))
    return output


class Siglip2Encoder:
    """Frozen SigLIP2 dual encoder, loaded only in the GPU environment.

    Imports are intentionally lazy: the dependency-free selector and its tests
    remain usable on a laptop without PyTorch or Transformers.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        *,
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int = 16,
        local_files_only: bool = False,
        revision: str = "main",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Siglip2Encoder requires torch and transformers; install the GPU "
                "dependencies described in docs/GPU_RUNBOOK.md"
            ) from error

        self._torch = torch
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            self.dtype_name = "bfloat16" if self.device.startswith("cuda") else "float32"
        else:
            self.dtype_name = dtype
        if self.dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16")
        torch_dtype = getattr(torch, self.dtype_name)

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            revision=revision,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=torch_dtype,
            local_files_only=local_files_only,
            revision=revision,
        ).to(self.device)
        self.model.eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision

    @property
    def fingerprint(self) -> str:
        return f"{self.model_id}|{self.dtype_name}|{self.resolved_revision}|slow-processor-v1"

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _as_normalized_lists(self, features: Any) -> list[list[float]]:
        torch = self._torch
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        return features.detach().cpu().tolist()

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        clean = [str(text).strip() for text in texts]
        if not clean or any(not text for text in clean):
            raise ValueError("texts must contain at least one non-empty string")
        inputs = self.processor(
            text=clean,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            features = self.model.get_text_features(**self._to_device(inputs))
        return self._as_normalized_lists(features)

    def encode_images(self, images: Sequence[Any]) -> list[list[float]]:
        if not images:
            return []
        output: list[list[float]] = []
        for start in range(0, len(images), self.batch_size):
            batch = list(images[start : start + self.batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            with self._torch.inference_mode():
                features = self.model.get_image_features(**self._to_device(inputs))
            output.extend(self._as_normalized_lists(features))
        return output
