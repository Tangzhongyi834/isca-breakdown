def is_attention_softmax(op, scopes):
    return "attention" in scopes and op in {
        "aten.softmax.int", "aten._softmax.default", "aten._safe_softmax.default"}
