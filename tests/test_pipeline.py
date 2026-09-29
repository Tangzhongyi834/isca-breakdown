"""Full CLI/output smoke test with explicitly mocked data, never paper data."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from profile_llm import parse_args, run
from llm_runtime_profile.data.wikitext2 import token_digest
from llm_runtime_profile.profiler.result_aggregator import read_csv


@unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1" and torch.cuda.is_available(), "Opt-in CUDA integration test")
class PipelineTests(unittest.TestCase):
    def test_cached_precision_comparison_outputs_and_rerun(self):
        with tempfile.TemporaryDirectory(prefix="llm-cli-test-") as directory:
            root = Path(directory)
            checkpoint = root / "test-fixture-llama"
            model = AutoModelForCausalLM.from_config(LlamaConfig(
                vocab_size=128, hidden_size=64, intermediate_size=128,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))
            model.save_pretrained(checkpoint)
            del model
            samples = torch.arange(8192).remainder(128).reshape(2, 4096)
            metadata = {"token_sha256": token_digest(samples), "dataset": "TEST FIXTURE ONLY",
                        "tokenizer": "test fixture"}
            output = root / "test-output"
            with patch("llm_runtime_profile.data.wikitext2.prepare_samples", return_value=(samples, metadata)), redirect_stdout(io.StringIO()):
                for precision in ("fp16", "w8a8", "fp16"):
                    args = parse_args(["--model", str(checkpoint), "--precision", precision,
                                       "--seq-lens", "16", "--num-samples", "2", "--warmup", "1",
                                       "--repeats", "2", "--output-dir", str(output), "--local-files-only"])
                    run(args)
            raw = read_csv(output / "raw/runtime_samples.csv")
            self.assertEqual(len(raw), 4)  # Rerun replaces the active configuration; no pooling.
            self.assertEqual({row["precision"] for row in raw}, {"FP16", "W8A8"})
            self.assertEqual(len({row["fairness_id"] for row in raw}), 1)
            summary = read_csv(output / "summary/runtime_by_length.csv")
            self.assertEqual(len(summary), 2)
            self.assertTrue((output / "summary/runtime_statistics.csv").exists())
            self.assertTrue((output / "summary/quantization_overhead.csv").exists())
            self.assertTrue((output / "summary/figure1_data.csv").exists())
            environment = json.loads((output / "metadata/environment.json").read_text())
            self.assertEqual(len(environment["runs"]), 2)
            for env in environment["runs"]:
                self.assertTrue(env["backend_evidence"]["gemm_kernels"])
                self.assertEqual(env["dataset"]["dataset"], "TEST FIXTURE ONLY")
            self.assertEqual(len(list(output.glob("runs/*/traces/*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
