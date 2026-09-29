from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from profile_llm import parse_args
from llm_runtime_profile.data.wikitext2 import construct_windows, prepare_samples, token_digest
from llm_runtime_profile.instrumentation.scopes import current_scopes, wrap_forward
from llm_runtime_profile.profiler.kernel_classifier import classify_operation
from llm_runtime_profile.profiler.result_aggregator import build_summaries, reduce_measurements, statistics_for
from llm_runtime_profile.profiler.runtime_profiler import derive_metrics, LEAF_CATEGORIES
from llm_runtime_profile.profiler.trace_analysis import analyze_trace, UnresolvedProfile
from llm_runtime_profile.quantization.base import BackendUnavailable
from llm_runtime_profile.quantization.w4a4 import W4A4Backend


def measurement(value):
    return derive_metrics({**{f"{key}_ms": value for key in LEAF_CATEGORIES},
                           "physical_total_ms": value * 30, "profiled_physical_ms": value * 35,
                           "gpu_kernel_total_ms": value * len(LEAF_CATEGORIES),
                           "gpu_activity_total_ms": value * len(LEAF_CATEGORIES), "consistency_error_pct": 0})


class DatasetTests(unittest.TestCase):
    def test_windows_prefix_offset_and_short_input(self):
        samples = construct_windows(list(range(9000)), 2, 13)
        self.assertEqual(samples.shape, (2, 4096))
        self.assertEqual(samples[1, 0].item(), 4109)
        for length in (512, 1024, 2048, 4096):
            self.assertEqual(samples[0, :length].tolist(), list(range(13, 13 + length)))
        with self.assertRaises(ValueError):
            construct_windows(list(range(8191)), 2)

    def test_cache_identity_and_corruption(self):
        class Tokenizer:
            name_or_path = "test/tokenizer"
            init_kwargs = {"_commit_hash": "test"}

            def __call__(self, text, **kwargs):
                self.text = text
                assert kwargs == {"add_special_tokens": False, "return_attention_mask": False}
                return {"input_ids": list(range(8192))}

        class Dataset(list):
            _fingerprint = "fixture"

        tokenizer = Tokenizer()
        dataset = Dataset([{"text": ""}, {"text": " Title "}, {"text": "  "}, {"text": "Text!"}])
        with tempfile.TemporaryDirectory() as directory:
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch("datasets.load_dataset", return_value=dataset):
                a, metadata = prepare_samples("fixture/a", "llama", 1, cache_dir=directory)
                self.assertEqual(tokenizer.text, " Title \n\nText!")
                b, second = prepare_samples("fixture/b", "llama", 1, cache_dir=directory)
                self.assertNotEqual(metadata["cache_path"], second["cache_path"])
            with patch("datasets.load_dataset", side_effect=AssertionError("Cache must avoid re-download")):
                cached, _ = prepare_samples("fixture/a", "llama", 1, cache_dir=directory)
            self.assertTrue(torch.equal(a, cached))
            self.assertEqual(token_digest(a), token_digest(b))
            payload = torch.load(metadata["cache_path"], weights_only=True)
            payload["samples"][0, 0] += 1
            torch.save(payload, metadata["cache_path"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_samples("fixture/a", "llama", 1, cache_dir=directory)


class ClassificationTests(unittest.TestCase):
    def test_exclusive_scopes(self):
        cases = [(("attention", "linear_module"), "aten.mm.default", "linear"),
                 (("attention",), "aten.matmul.default", "attention_matmul"),
                 (("attention", "rope"), "aten.bmm.default", "rope"),
                 (("attention", "linear_module", "quantize"), "aten.round.default", "quantize"),
                 (("attention",), "aten.softmax.int", "softmax"),
                 ((), "aten.softmax.int", "other_compute"),
                 (("mlp",), "aten.silu.default", "silu"),
                 (("mlp",), "aten.mul.Tensor", "other_compute"),
                 (("rmsnorm",), "aten.to.dtype", "rmsnorm"),
                 ((), "aten.to.dtype", "conversion")]
        for scopes, op, expected in cases:
            self.assertEqual(classify_operation(op, scopes), expected)
        with self.assertRaisesRegex(RuntimeError, "Fused attention"):
            classify_operation("aten._scaled_dot_product_flash_attention.default", ("attention",))

    def test_forward_restored_after_exception(self):
        module = torch.nn.Identity()
        with self.assertRaises(ValueError):
            with wrap_forward(module, "rope"):
                self.assertTrue("forward" in module.__dict__)
                raise ValueError("fixture")
        self.assertNotIn("forward", module.__dict__)
        self.assertEqual(current_scopes(), ())

    def test_kernel_attribution_counts_each_kernel_once(self):
        trace = {"traceEvents": [
            {"ph": "X", "pid": 1, "tid": 1, "ts": 0, "dur": 100, "name": "llm_profile::linear"},
            {"ph": "X", "cat": "cpu_op", "pid": 1, "tid": 1, "ts": 1, "dur": 80,
             "name": "aten::mm", "args": {"External id": 1}},
            {"ph": "X", "cat": "kernel", "ts": 200, "dur": 20, "name": "test_gemm", "args": {"External id": 1}},
        ]}
        totals, total, inventory = analyze_trace(trace, "fp16")
        self.assertEqual(totals, {"linear": .02})
        self.assertEqual(total, .02)
        self.assertEqual(inventory[0]["calls"], 1)
        trace["traceEvents"].append({"ph": "X", "cat": "gpu_memset", "ts": 190, "dur": 2,
                                     "name": "Memset (Device)", "args": {"External id": 1}})
        totals, total, inventory = analyze_trace(trace, "fp16")
        self.assertEqual(totals, {"linear": .02, "conversion": .002})
        self.assertEqual(total, .022)
        with self.assertRaisesRegex(UnresolvedProfile, "UNRESOLVED_FUSED"):
            analyze_trace(trace, "w8a8")
        trace["traceEvents"][-1]["args"]["External id"] = 999
        with self.assertRaisesRegex(UnresolvedProfile, "Uncorrelated"):
            analyze_trace(trace, "fp16")


class AggregationTests(unittest.TestCase):
    def test_component_medians_remain_additive(self):
        rows = [measurement(value) for value in (1, 2, 100)]
        rows[0]["linear_ms"], rows[1]["linear_ms"], rows[2]["linear_ms"] = 100, 1, 2
        reduced = reduce_measurements(rows)
        self.assertEqual(reduced["linear_ms"], 2)
        self.assertEqual(reduced["nonlinear_ms"], 8)
        self.assertEqual(reduced["compute_total_ms"], 14)
        self.assertAlmostEqual(sum(reduced[f"{name}_pct"] for name in ("linear", "nonlinear", "other")), 100)
        self.assertEqual(reduced["quantization_overhead_ms"], 12)

    def test_statistics_and_paper_gate(self):
        self.assertEqual(statistics_for([0., 10.])["p10"], 1)
        self.assertEqual(statistics_for([0., 10.])["p90"], 9)
        identity = dict(model="fixture", precision="FP16", seq_len=512, measurement_kind="measured",
                        validation_status="passed", fairness_id="same", token_sha256="tokens", run_id="one")
        rows = [{**identity, "sample_id": i, **measurement(i + 1)} for i in range(3)]
        summary, overhead, figures, stats = build_summaries(rows)
        self.assertEqual(summary[0]["num_samples"], 3)
        self.assertEqual(overhead[0]["total_quantization_overhead_ms"], 12)
        self.assertEqual(figures[0]["nonlinear_ms"], 8)
        self.assertTrue(stats)
        with self.assertRaisesRegex(ValueError, "Projected"):
            build_summaries([{**rows[0], "measurement_kind": "projected"}])
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            build_summaries([rows[0], {**rows[1], "fairness_id": "different"}])


class CLITests(unittest.TestCase):
    def test_defaults_and_invalid_inputs(self):
        args = parse_args(["--model", "fixture"])
        self.assertEqual((args.seq_lens, args.num_samples, args.warmup, args.repeats), ([512, 1024, 2048, 4096], 32, 5, 3))
        for options in (["--seq-lens", "4097"], ["--seq-lens", "512", "512"], ["--batch-size", "2"], ["--phase", "decode"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--model", "fixture", *options])

    def test_w4a4_never_falls_back(self):
        with self.assertRaisesRegex(BackendUnavailable, "True W4A4 CUDA backend is unavailable."):
            W4A4Backend().validate()


if __name__ == "__main__":
    unittest.main()
