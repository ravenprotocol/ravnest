from .llama import LlamaForCausalLMPipelineStage

def get_pipeline_stage_submod(model, stage_idx, node_type, start_index, end_index):
    if model.__class__.__name__ == 'LlamaForCausalLM':
        return LlamaForCausalLMPipelineStage(model, stage_idx, node_type, start_index, end_index)