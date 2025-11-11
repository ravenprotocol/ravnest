from .llama import LlamaForCausalLMPipelineStage
# from .qwen_2 import Qwen2ForCausalLMPipelineStage

name_stage_mapping = {
    'LlamaForCausalLM':LlamaForCausalLMPipelineStage,
    # 'Qwen2ForCausalLM':Qwen2ForCausalLMPipelineStage
}

def get_inference_pipeline_stage_submod(model, stage_idx, node_type, start_index, end_index):
    model_class_name = model.__class__.__name__
    pipeline_stage = name_stage_mapping.get(model_class_name, None)
    return pipeline_stage(model, stage_idx, node_type, start_index, end_index)