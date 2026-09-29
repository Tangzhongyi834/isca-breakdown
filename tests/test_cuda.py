"""Opt-in real GPU smoke tests. Random fixtures are NOT experimental results."""

import os
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, MistralConfig

from llm_runtime_profile.environment import gpu_lock
from llm_runtime_profile.models import get_adapter
from llm_runtime_profile.profiler.runtime_profiler import RuntimeProfiler
from llm_runtime_profile.quantization import create_backend
from llm_runtime_profile.quantization.w8a8 import W8A8Linear


@unittest.skipUnless(os.environ.get("RUN_CUDA_TESTS") == "1" and torch.cuda.is_available(), "Set RUN_CUDA_TESTS=1 on a CUDA GPU")
class CUDATests(unittest.TestCase):
    def test_huggingface_two_architectures_and_precisions(self):
        device = torch.device("cuda:0")
        with gpu_lock(device), tempfile.TemporaryDirectory(prefix="llm-runtime-test-") as directory:
            for family, config_type in (("llama", LlamaConfig), ("mistral", MistralConfig)):
                config = config_type(vocab_size=128, hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                     max_position_embeddings=4096)
                checkpoint = Path(directory) / family
                torch.manual_seed(7)
                fixture = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
                fixture.save_pretrained(checkpoint)
                del fixture
                for precision in ("fp16", "w8a8"):
                    with self.subTest(family=family, precision=precision):
                        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16,
                                                                   attn_implementation="eager").to(device).eval()
                        adapter = get_adapter(model)
                        backend = create_backend(precision, device)
                        backend.validate()
                        backend.quantize_model(model)
                        runner = RuntimeProfiler(model, adapter, backend, device)
                        # Small shape exercises integer padding; longer lengths exercise eager attention.
                        lengths = (16, 512, 1024, 2048, 4096) if os.environ.get("RUN_FULL_LENGTH_TESTS") == "1" else (16, 512)
                        for length in lengths:
                            input_ids = torch.arange(length, device=device).remainder(128).unsqueeze(0)
                            runner.warmup(input_ids, 1)
                            row, evidence = runner.measure(input_ids)
                            for category in ("linear", "softmax", "rope", "silu", "rmsnorm", "attention_matmul"):
                                self.assertGreater(row[category + "_ms"], 0)
                            self.assertAlmostEqual(row["compute_total_ms"] + row["quantization_overhead_ms"], row["gpu_activity_total_ms"])
                            self.assertAlmostEqual(sum(row[name + "_pct"] for name in ("linear", "nonlinear", "other")), 100)
                            if precision == "w8a8":
                                self.assertGreater(row["quantize_ms"], 0)
                                self.assertGreater(row["dequantize_ms"], 0)
                                self.assertTrue(all(k["operator"] == "aten::_int_mm" for k in evidence["kernels"] if k["category"] == "linear"))
                            self.assertTrue(backend.evidence["gemm_kernels"])
                            self.assertNotIn("forward", model.model.layers[0].self_attn.__dict__)
                        del runner, adapter, model
                        torch.cuda.empty_cache()

    def test_unaligned_int8_linear_dtypes_and_padding(self):
        with gpu_lock("cuda:0"), torch.inference_mode():
            original = torch.nn.Linear(35, 23, bias=True, device="cuda", dtype=torch.float16)
            module = W8A8Linear(original)
            result = module(torch.randn(1, 7, 35, device="cuda", dtype=torch.float16))
            self.assertEqual(result.shape, (1, 7, 23))
            self.assertEqual(module.weight_int8.dtype, torch.int8)
            self.assertEqual(result.dtype, torch.float16)
            self.assertTrue(torch.isfinite(result).all().item())


if __name__ == "__main__":
    unittest.main()
