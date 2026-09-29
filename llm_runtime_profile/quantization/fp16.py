import torch
from torch import nn

from .base import QuantBackend, replace_linears


class FP16Linear(nn.Module):
    """Explicit bias-free GEMM; a model's optional bias is a separate operation."""

    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features = original.in_features
        self.out_features = original.out_features

    def forward(self, x):
        output = torch.mm(x.reshape(-1, self.in_features), self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*x.shape[:-1], self.out_features)


class FP16Backend(QuantBackend):
    precision = "fp16"
    weight_precision = "FP16"
    activation_precision = "FP16"

    def backend_name(self):
        return "PyTorch torch.mm / CUDA cuBLAS FP16"

    def validate(self):
        self.check_cuda()
        self.evidence = {"operator": "aten::mm", "gemm_kernels": []}
        return self.evidence

    def quantize_model(self, model):
        return replace_linears(model, FP16Linear)
