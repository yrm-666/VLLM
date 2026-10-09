from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from molmo2_frame_selector.encoders import Siglip2Encoder, cosine_similarities, l2_normalize


class EncoderMathTest(unittest.TestCase):
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
