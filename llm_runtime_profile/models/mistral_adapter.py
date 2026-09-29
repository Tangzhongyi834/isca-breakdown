from transformers.models.mistral import modeling_mistral as modeling

from .base_adapter import ModelAdapter


class MistralAdapter(ModelAdapter):
    def __init__(self, model):
        super().__init__(model, modeling, modeling.MistralAttention, modeling.MistralMLP,
                         modeling.MistralRMSNorm, modeling.MistralRotaryEmbedding)
