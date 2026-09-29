from .linear_gemm import is_gemm


def is_attention_matmul(op, scopes):
    return "attention" in scopes and "linear_module" not in scopes and is_gemm(op)
