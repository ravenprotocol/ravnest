from .llama_split_spec import LlamaSplitSpec, LlamaForCausalLMSplitSpec

def get_split_spec(model):
    if model.__class__.__name__ == 'LlamaModel':
        return LlamaSplitSpec
    elif model.__class__.__name__ == 'LlamaForCausalLM':
        return LlamaForCausalLMSplitSpec