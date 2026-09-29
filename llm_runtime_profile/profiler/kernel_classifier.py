"""Exclusive classification from architectural context and ATen operation."""

import re

from torch.profiler import record_function
from torch.utils._python_dispatch import TorchDispatchMode

from ..instrumentation.attention_matmul import is_attention_matmul
from ..instrumentation.linear_gemm import is_gemm
from ..instrumentation.quantization import QUANT_CATEGORIES
from ..instrumentation.scopes import current_scopes
from ..instrumentation.silu import is_silu
from ..instrumentation.softmax import is_attention_softmax

PREFIX = "llm_profile::"
GEMM_KERNEL = re.compile(r"gemm|mma|cutlass|cublas.*(compute|kernel)", re.I)
CONVERSION_OPS = {"aten._to_copy.default", "aten.copy_.default", "aten.clone.default",
                  "aten.to.dtype", "aten.to.device", "aten.to.dtype_layout",
                  "aten.contiguous.default", "aten.reshape.default"}


def classify_operation(op, scopes):
    for name in reversed(scopes):
        if name in QUANT_CATEGORIES or name in ("rope", "rmsnorm"):
            return name
        if name == "linear_module":
            if op == "aten.addmm.default":
                raise RuntimeError("Fused bias/GEMM is not supported; split bias from GEMM")
            if op in CONVERSION_OPS:
                return "conversion"
            return "linear" if is_gemm(op) else "other_compute"
    if "scaled_dot_product" in op or "flash_attention" in op:
        raise RuntimeError("Fused attention cannot be separated; require eager attention")
    if is_attention_matmul(op, scopes):
        return "attention_matmul"
    if is_attention_softmax(op, scopes):
        return "softmax"
    if is_silu(op, scopes):
        return "silu"
    # Standalone dtype/layout conversions outside complete nonlinear scopes.
    if op in CONVERSION_OPS:
        return "conversion"
    return "other_compute"


class OperationInstrumentation(TorchDispatchMode):
    def __init__(self, timer):
        super().__init__()
        self.timer = timer

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        category = classify_operation(str(func), current_scopes())
        with record_function(PREFIX + category), self.timer.measure(category):
            return func(*args, **(kwargs or {}))
