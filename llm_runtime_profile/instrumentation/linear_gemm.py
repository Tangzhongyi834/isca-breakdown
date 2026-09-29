GEMM_OPS = {"aten.mm.default", "aten.bmm.default", "aten.matmul.default", "aten._int_mm.default"}


def is_gemm(op):
    return op in GEMM_OPS
