from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from molmo2_frame_selector.encoders import Siglip2Encoder, cosine_similarities, l2_normalize
from molmo2_frame_selector.config import SelectorConfig
from molmo2_frame_selector.pipeline import QueryAwareSelectionPipeline
from molmo2_frame_selector.selector import QueryAwareFrameSelector


class AmbiguousArray:
    """Reproduce ndarray truth-value semantics without depending on NumPy."""
    def __init__(self, rows):
        self.rows = rows

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return self.rows[key]


class EncoderMathTest(unittest.TestCase):
    def fake_image_encoder(self):
        encoder = Siglip2Encoder.__new__(Siglip2Encoder)
        encoder.batch_size = 2
        encoder._torch = SimpleNamespace(inference_mode=nullcontext)
        encoder.device = "cpu"

        class PixelBatch:
            def __init__(self, rows):
                self.rows = rows

            def to(self, device):
                return self

        encoder.processor = lambda *, images, return_tensors: {"pixel_values": PixelBatch(images)}
        encoder.model = SimpleNamespace(get_image_features=lambda *, pixel_values: pixel_values.rows)
        encoder._as_normalized_lists = lambda features: features
        encoder.encode_texts = lambda texts: [[1.0, 0.0] for _ in texts]
        return encoder

    def test_ndarray_style_image_batch_and_empty_batch(self):
        encoder = self.fake_image_encoder()
        rows = [[1, 0], [0, 1], [0.8, 0.2]]
        self.assertEqual(encoder.encode_images(AmbiguousArray(rows)), rows)
        self.assertEqual(encoder.encode_images(AmbiguousArray([])), [])

    def test_cold_pipeline_accepts_ndarray_style_frames(self):
        pipeline = QueryAwareSelectionPipeline(
            self.fake_image_encoder(), QueryAwareFrameSelector(SelectorConfig(2, 3, temporal_bins=2)))
        result = pipeline.select("query", AmbiguousArray([[1, 0], [0, 1], [0.8, 0.2]]),
                                 timestamps=(0.0, 1.0, 2.0), frame_indices=(0, 10, 20),
                                 time_range=(0.0, 3.0))
        self.assertEqual(result.cache_hits, 0)
        self.assertEqual(result.cache_misses, 3)
        self.assertEqual(len(result.selected_frames), 2)

    def test_l2_normalize(self) -> None:
        self.assertEqual(l2_normalize([3.0, 4.0]), [0.6, 0.8])

    def test_cosine_similarities(self) -> None:
        scores = cosine_similarities(
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        )
        self.assertEqual(scores, [1.0, 0.0, -1.0])

    def test_zero_embedding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            l2_normalize([0.0, 0.0])

    def test_processor_does_not_force_slow_gemma_tokenizer(self) -> None:
        class FakeImageProcessor:
            pass

        class FakeFastTokenizer:
            pass

        processor = SimpleNamespace(image_processor=FakeImageProcessor(), tokenizer=FakeFastTokenizer())
        auto_processor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
        model = Mock()
        model.to.return_value = model
        model.config = SimpleNamespace(_commit_hash="fixed-revision")
        auto_model = SimpleNamespace(from_pretrained=Mock(return_value=model))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float32="float32")
        transformers = SimpleNamespace(AutoProcessor=auto_processor, AutoModel=auto_model)
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            encoder = Siglip2Encoder(local_files_only=True, revision="fixed-revision")
        kwargs = auto_processor.from_pretrained.call_args.kwargs
        self.assertNotIn("use_fast", kwargs)
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(kwargs["revision"], "fixed-revision")
        self.assertIn("FakeImageProcessor|FakeFastTokenizer", encoder.fingerprint)
        model.eval.assert_called_once()
