from pathlib import Path
from ...strings import *
from ...lazy_init.lazy_context import LazyInitContext
from ...lazy_init.interface import get_pretrained_path, set_pretrained_path
from ..utils import load_stage_weights_from_checkpoint, set_tensors_to_none, load_state_dict
from .base_split_spec import BaseSplitSpec

class Qwen2SplitSpec(BaseSplitSpec):
    def get_stage_layers(self):
        print(self.model.__class__.__name__)
        if self.model.__class__.__name__ == "Qwen2Model":
            module = self.model
        else:
            module = self.model.model

        held_layers = []
        if self.node_type == NodeTypes.ROOT:
            held_layers.append(module.embed_tokens)
        held_layers.append(module.rotary_emb)
        start_idx, end_idx = self.get_stage_layer_indices(len(module.layers))
        held_layers.extend(module.layers[start_idx:end_idx])
        if self.node_type == NodeTypes.LEAF:
            held_layers.append(module.norm)
        return held_layers
    
    def remove_non_stage_layers(self):
        stage_layers = self.get_stage_layers()
        set_tensors_to_none(self.model, exclude=set(stage_layers))
        
    def configure_stage_model(self):
        self.remove_non_stage_layers()
        LazyInitContext.materialize(self.model)

        pretrained_path = get_pretrained_path(self.model)
        if pretrained_path:
            if '.index.json' in pretrained_path:
                print('Pretrained path: ', pretrained_path)
                load_stage_weights_from_checkpoint(self.model, Path(pretrained_path))
                # set_pretrained_path(self.model, None)
            else:
                state_dict = load_state_dict(pretrained_path)
                self.model.load_state_dict(state_dict, strict=False)

class Qwen2ForCausalLMSplitSpec(Qwen2SplitSpec):
    def get_stage_layers(self):
        held_layers = super().get_stage_layers()
        if self.node_type == NodeTypes.LEAF:
            held_layers.append(self.model.lm_head)
            if self.tie_weight_check():
                held_layers.append(self.model.model.embed_tokens)
        return held_layers

    def get_shared_params(self):
        if self.tie_weight_check():
            return {
                0: self.model.model.embed_tokens.weight,
                self.num_stages - 1: self.model.lm_head.weight
            }
        return None

    def tie_weight_check(self):
        input_embedding = self.model.get_input_embeddings()
        output_embedding = self.model.get_output_embeddings()
        return (
            input_embedding is not None
            and output_embedding is not None
            and id(input_embedding.weight) == id(output_embedding.weight)
        )