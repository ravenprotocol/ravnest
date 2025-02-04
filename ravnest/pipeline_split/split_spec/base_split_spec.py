import numpy as np

class BaseSplitSpec():
    def __init__(self, stage, node_type, model, num_stages):
        self.stage = stage
        self.node_type = node_type
        self.model = model
        self.num_stages = num_stages

    def get_stage_layer_indices(self, num_layers):
        layers_per_stage = self.get_layers_per_stage(num_layers)
        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)

        start_idx = num_layers_per_stage_accumulated[self.stage]
        end_idx = num_layers_per_stage_accumulated[self.stage + 1]
        return [start_idx, end_idx]
    
    def get_layers_per_stage(self, num_layers):
        quotient = num_layers // self.num_stages
        remainder = num_layers % self.num_stages

        layers_per_stage = [quotient] * self.num_stages
        if remainder > 0:
            start_position = self.num_stages // 2 - remainder // 2
            for i in range(start_position, start_position + remainder):
                layers_per_stage[i] += 1
        
        return layers_per_stage
    
    def configure_stage_model(self):
        '''
        Configures model for this pipeline stage by retaining only the required layers
        '''
        ...
    
    # def get_intermediate_input_keys(self):
    #     return ("hidden_states",)