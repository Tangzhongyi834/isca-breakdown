from contextlib import ExitStack, contextmanager

from .scopes import wrap_forward


@contextmanager
def instrument_rmsnorm(modules):
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(wrap_forward(module, "rmsnorm"))
        yield
