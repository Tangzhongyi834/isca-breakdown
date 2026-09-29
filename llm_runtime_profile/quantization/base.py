from abc import ABC, abstractmethod

import torch


class BackendUnavailable(RuntimeError):
    pass


class QuantBackend(ABC):
    precision = None
    weight_precision = None
    activation_precision = None

    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)
        self.evidence = {}

    @abstractmethod
    def quantize_model(self, model):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def backend_name(self):
        pass

    def check_cuda(self):
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise BackendUnavailable("This framework requires a CUDA GPU; CPU timing is not paper data")

    def describe(self):
        print(f"Quantization Mode    : {self.precision.upper()}\n"
              f"Weight Precision     : {self.weight_precision}\n"
              f"Activation Precision : {self.activation_precision}\n"
              f"GEMM Backend         : {self.backend_name()}\n"
              f"GEMM Kernel          : {', '.join(self.evidence.get('gemm_kernels', []))}")


def replace_linears(model, factory):
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)]
    for name, module in modules:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, factory(module))
    return model
