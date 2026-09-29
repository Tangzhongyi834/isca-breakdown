from contextlib import ExitStack, contextmanager, nullcontext

import torch

from ..instrumentation.rmsnorm import instrument_rmsnorm
from ..instrumentation.rope import instrument_rope
from ..instrumentation.scopes import wrap_forward


class ModelAdapter:
    def __init__(self, model, modeling, attention_class, mlp_class, norm_class, rotary_class):
        self.model = model
        self.modeling = modeling
        self.attention_class = attention_class
        self.mlp_class = mlp_class
        self.norm_class = norm_class
        self.rotary_class = rotary_class
        if model.config._attn_implementation != "eager":
            raise ValueError("attn_implementation must be eager")
        if model.config.hidden_act != "silu":
            raise ValueError("This experiment requires the SiLU LLaMA/Mistral architecture")
        if getattr(model.config, "pretraining_tp", 1) != 1:
            raise ValueError("pretraining_tp must be 1; tensor-parallel functional projections are not supported")
        # Capture type-based identity before any backend replaces nn.Linear.
        self.linear_names = tuple(name for name, module in model.named_modules()
                                  if isinstance(module, torch.nn.Linear))
        if not self.linear_names:
            raise ValueError("No linear modules found")

    def get_linear_modules(self):
        return {name: self.model.get_submodule(name) for name in self.linear_names}

    def get_rmsnorm_modules(self):
        return [module for module in self.model.modules() if isinstance(module, self.norm_class)]

    def instrument_rope(self):
        return instrument_rope(self.model, self.modeling, self.rotary_class)

    @contextmanager
    def instrument_silu(self):
        with ExitStack() as stack:
            for module in self.model.modules():
                if isinstance(module, self.mlp_class):
                    stack.enter_context(wrap_forward(module, "mlp"))
            yield

    def instrument_softmax(self):
        # Actual softmax selection is done by ATen dispatch within attention scope.
        return nullcontext()

    @contextmanager
    def instrument_attention_matmul(self):
        with ExitStack() as stack:
            for module in self.model.modules():
                if isinstance(module, self.attention_class):
                    stack.enter_context(wrap_forward(module, "attention"))
            yield

    @contextmanager
    def instrument(self):
        with ExitStack() as stack:
            stack.enter_context(self.instrument_rope())
            stack.enter_context(self.instrument_silu())
            stack.enter_context(self.instrument_softmax())
            stack.enter_context(self.instrument_attention_matmul())
            stack.enter_context(instrument_rmsnorm(self.get_rmsnorm_modules()))
            for module in self.get_linear_modules().values():
                stack.enter_context(wrap_forward(module, "linear_module"))
            yield
