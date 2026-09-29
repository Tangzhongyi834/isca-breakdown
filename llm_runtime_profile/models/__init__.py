def get_adapter(model):
    from .llama_adapter import LlamaAdapter
    from .mistral_adapter import MistralAdapter

    adapters = {"llama": LlamaAdapter, "mistral": MistralAdapter}
    try:
        return adapters[model.config.model_type](model)
    except KeyError as exc:
        raise ValueError(f"Unsupported model_type: {model.config.model_type}") from exc
