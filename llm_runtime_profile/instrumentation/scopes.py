from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_scopes = ContextVar("llm_profile_scopes", default=())


@contextmanager
def scope(name):
    token = _scopes.set((*_scopes.get(), name))
    try:
        yield
    finally:
        _scopes.reset(token)


def current_scopes():
    return _scopes.get()


@contextmanager
def wrap_forward(module, name):
    # Restore the instance dictionary, not just a captured bound method.
    existed = "forward" in module.__dict__
    previous = module.__dict__.get("forward")
    original = module.forward

    @wraps(original)
    def forward(*args, **kwargs):
        with scope(name):
            return original(*args, **kwargs)

    module.forward = forward
    try:
        yield
    finally:
        if existed:
            module.forward = previous
        else:
            del module.forward


@contextmanager
def wrap_function(owner, attribute, name):
    original = getattr(owner, attribute)

    @wraps(original)
    def wrapped(*args, **kwargs):
        with scope(name):
            return original(*args, **kwargs)

    setattr(owner, attribute, wrapped)
    try:
        yield
    finally:
        setattr(owner, attribute, original)
