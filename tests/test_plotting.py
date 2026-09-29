import importlib.util
from pathlib import Path
import tempfile
import unittest

from llm_runtime_profile.profiler.result_aggregator import write_csv


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
class PlottingTests(unittest.TestCase):
    def test_figures_and_rejection_of_projected_data(self):
        from plotting.plot_runtime_breakdown import main as breakdown
        from plotting.plot_nonlinear_fraction import main as nonlinear
        from plotting.plot_cross_model import main as cross_model
        # Explicit test fixtures exist only in a temporary directory.
        rows = [dict(model=model, precision=precision, seq_len=length, measurement_kind="measured",
                     validation_status="passed", fairness_id=model, token_sha256=model,
                     linear_ms=2., nonlinear_ms=1., other_ms=1., compute_total_ms=4.,
                     linear_pct=50., nonlinear_pct=25., other_pct=25.)
                for model in ("test-fixture/llama", "test-fixture/mistral")
                for precision in ("FP16", "W8A8") for length in (512, 1024, 2048, 4096)]
        with tempfile.TemporaryDirectory(prefix="llm-plot-test-") as directory:
            path = Path(directory) / "fixture.csv"
            write_csv(path, rows)
            options = ["--input", str(path), "--output-dir", directory, "--allow-incomplete"]
            for plot in (breakdown, nonlinear, cross_model):
                plot(options)
            self.assertEqual(len(list(Path(directory).glob("*.pdf"))), 3)
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 3)
            for pdf in Path(directory).glob("*.pdf"):
                self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                nonlinear(options[:-1])
            rows[0]["measurement_kind"] = "projected"
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "projected"):
                breakdown(options)


if __name__ == "__main__":
    unittest.main()
