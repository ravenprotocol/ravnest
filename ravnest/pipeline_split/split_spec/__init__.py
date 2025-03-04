from .llama_split_spec import LlamaSplitSpec, LlamaForCausalLMSplitSpec
from .qwen_2_split_spec import Qwen2SplitSpec, Qwen2ForCausalLMSplitSpec

name_spec_mapping = {
    'LlamaModel':LlamaSplitSpec,
    'LlamaForCausalLM':LlamaForCausalLMSplitSpec,
    'Qwen2Model':Qwen2SplitSpec,
    'Qwen2ForCausalLM':Qwen2ForCausalLMSplitSpec
}

def get_split_spec(model):
    model_class_name = model.__class__.__name__
    return name_spec_mapping.get(model_class_name)