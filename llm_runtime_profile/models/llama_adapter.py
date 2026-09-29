from transformers.models.llama import modeling_llama as modeling

from .base_adapter import ModelAdapter


class LlamaAdapter(ModelAdapter):
    def __init__(self, model):
        super().__init__(model, modeling, modeling.LlamaAttention, modeling.LlamaMLP,
                         modeling.LlamaRMSNorm, modeling.LlamaRotaryEmbedding)
