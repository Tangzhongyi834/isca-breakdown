from contextlib import ExitStack, contextmanager

from .scopes import wrap_forward, wrap_function


@contextmanager
def instrument_rope(model, modeling, rotary_class):
    with ExitStack() as stack:
        for module in model.modules():
            if isinstance(module, rotary_class):
                stack.enter_context(wrap_forward(module, "rope"))
        stack.enter_context(wrap_function(modeling, "apply_rotary_pos_emb", "rope"))
        yield
