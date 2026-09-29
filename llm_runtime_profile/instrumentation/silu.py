def is_silu(op, scopes):
    return "mlp" in scopes and op == "aten.silu.default"
