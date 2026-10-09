"""No-visual-input boundary tests and deterministic action pilot selection."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from molmo2_frame_selector.experiment import VideoQAExample, VideoQAExperiment
from molmo2_frame_selector.hf_runner import GenerationResult, Molmo2HFRunner
from molmo2_frame_selector.reporting import compare_jsonl_files


class Round2Test(unittest.TestCase):
    def test_shared_generation_reports_text_output_without_visual_grid(self):
        class Tokens:
            def __init__(self, length):
                self.shape = (1, length)

            def __getitem__(self, key):
                return Tokens(self.shape[-1] - key[1].start)

        runner = Molmo2HFRunner.__new__(Molmo2HFRunner)
        runner._torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                         inference_mode=nullcontext)
        runner._move_inputs = lambda inputs: inputs
        runner.model = SimpleNamespace(generate=Mock(return_value=Tokens(5)))
        runner.processor = SimpleNamespace(post_process_image_text_to_text=Mock(return_value=["B"]))
        result = runner._generate_inputs({"input_ids": Tokens(3)}, .1, 64, False)
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.visual_tokens, 0)
        self.assertEqual(result.text, "B")
        self.assertIsNone(result.ttft_seconds)
        self.assertFalse(runner.model.generate.call_args.kwargs["do_sample"])

    def test_comparison_rejects_empty_result_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contains no records"):
                compare_jsonl_files([path])

    def test_text_only_never_decodes_or_encodes_video_and_keeps_same_query(self):
        decoder, encoder, runner = Mock(), Mock(), Mock()
        runner.generate_text.return_value = GenerationResult("(B) cup", 10, 0, .1, None, .2, None, 3)
        example = VideoQAExample("x", Path("not-downloaded.mp4"), "What?", ("book", "cup"), "B")
        experiment = VideoQAExperiment(mode="text_only", decoder=decoder,
                                       molmo_runner=runner, encoder=encoder)
        result = experiment.run(example)
        decoder.decode.assert_not_called()
        encoder.encode_images.assert_not_called()
        runner.generate.assert_not_called()
        runner.generate_text.assert_called_once_with(result.query, max_new_tokens=128, measure_ttft=False)
        self.assertNotIn("gold", result.query.lower())
        self.assertIn("(A) book", result.query)
        self.assertTrue(result.record.correct)
        self.assertEqual(result.record.actual_visual_tokens, 0)
        self.assertEqual(result.selected_indices, ())

    def test_text_only_rejects_unexpected_visual_tokens_and_records_failure(self):
        runner = Mock()
        runner.generate_text.return_value = GenerationResult("A", 10, 81, .1, None, .2, None)
        experiment = VideoQAExperiment(mode="text_only", decoder=Mock(), molmo_runner=runner)
        result = experiment.run_safely(VideoQAExample("x", Path("x"), "Q", gold_label="A"))
        self.assertEqual(result.record.status, "error")
        self.assertFalse(result.record.correct)

    def test_real_text_adapter_sends_only_text_to_processor(self):
        runner = Molmo2HFRunner.__new__(Molmo2HFRunner)
        runner.processor = Mock()
        runner.processor.apply_chat_template.return_value = "templated Q"
        runner.processor.return_value = {"input_ids": "fake-tensor"}
        runner._generate_inputs = Mock(return_value="output")
        self.assertEqual(runner.generate_text("Q?", max_new_tokens=64), "output")
        messages = runner.processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages, [{"role": "user", "content": [{"type": "text", "text": "Q?"}]}])
        runner.processor.assert_called_once_with(text="templated Q", padding=True, return_tensors="pt")
        self.assertEqual(runner._generate_inputs.call_args.args[2:], (64, False))
        with self.assertRaises(ValueError):
            runner.generate_text("Q?", max_new_tokens=0)

    def action_module(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_action_validation.py"
        spec = importlib.util.spec_from_file_location("action_preparation_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_action_selection_keeps_source_order_and_unique_videos(self):
        module = self.action_module()
        rows = [{"video": name, "start": 1, "end": 2}
                for name in ("a.mp4", "a.mp4", "b.mp4", "c.mp4")]
        self.assertEqual(module.choose_indices(rows, 2), [0, 2])
        self.assertEqual(module.choose_indices(rows, 2, 2), [2, 3])
        with self.assertRaises(ValueError):
            module.choose_indices(rows, 4)

    def test_action_requires_bounds_and_rejects_path_escape(self):
        module = self.action_module()
        for row in [{"video": "x.mp4"}, {"video": "../x.mp4", "start": 1, "end": 2}]:
            with self.assertRaises(ValueError):
                module.choose_indices([row], 1)

    def test_batch_text_only_does_not_load_selector(self):
        # Same production batch CLI, not a separate unpaired evaluation script.
        from test_manifest_runner import load_runner, FakeRunner
        import json
        from contextlib import redirect_stdout
        import io
        module = load_runner()

        class TextRunner(FakeRunner):
            def generate_text(self, query, **kwargs):
                return GenerationResult("A", 10, 0, .01, None, .02, None, 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, output = root / "manifest.jsonl", root / "output.jsonl"
            manifest.write_text(json.dumps({"id": "x", "video": "missing.mp4", "question": "Q?",
                                           "options": ["yes", "no"], "answer": "A"}), encoding="utf-8")
            with patch("sys.argv", ["run_manifest.py", "--manifest", str(manifest), "--output", str(output),
                                    "--mode", "text_only", "--warmup", "1"]), \
                 patch.object(module, "Molmo2HFRunner", TextRunner), \
                 patch.object(module, "Siglip2Encoder") as encoder, redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 0)
            encoder.assert_not_called()
            row = json.loads(output.read_text())
            self.assertTrue(row["correct"])
            self.assertEqual(row["actual_visual_tokens"], 0)
            self.assertIsNone(row["selector_model_id"])


if __name__ == "__main__":
    unittest.main()
