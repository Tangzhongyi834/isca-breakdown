def create_backend(precision, device):
    from .fp16 import FP16Backend
    from .w8a8 import W8A8Backend
    from .w4a4 import W4A4Backend

    return {"fp16": FP16Backend, "w8a8": W8A8Backend, "w4a4": W4A4Backend}[precision](device)
