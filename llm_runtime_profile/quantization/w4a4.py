from .base import BackendUnavailable, QuantBackend


class W4A4Backend(QuantBackend):
    """Fail-closed backend slot. No weight-only or projected substitute is used.

    A future implementation must expose packing, true signed INT4 GEMM and
    dequantization separately, and extend kernel evidence validation accordingly.
    """

    precision = "w4a4"
    weight_precision = "INT4"
    activation_precision = "INT4"

    def backend_name(self):
        return "unavailable (requires an independently validated INT4 x INT4 CUDA backend)"

    def validate(self):
        raise BackendUnavailable("True W4A4 CUDA backend is unavailable.")

    def quantize_model(self, model):
        self.validate()
