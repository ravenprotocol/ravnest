from .strings import *
from .utils import *
import torch

class Compute():
    def __init__(self, model=None, optimizer=None, device=None, input_tensors=None, output_tensors=None, submod_file=None, criterion=None, input_template=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_tensors = input_tensors
        self.submod_file = submod_file
        self.criterion = criterion
        self.output_tensors = output_tensors
        self.input_template = input_template
    
    def middle_forward_compute(self, data, forward_pass_id):
        model_args = self.create_model_args(data, forward_pass_id=forward_pass_id, node_type = NodeTypes.MID)
        
        if not self.model.training:
            self.model.train()

        output = self.model(*model_args)
        return output
    
    def leaf_find_loss(self, data, targets):
        model_args = self.create_model_args(data, node_type=NodeTypes.LEAF)
        
        if not self.model.training:
            self.model.train()

        outputs = self.model(*model_args.values())
        
        loss = self.criterion(outputs, targets)
        
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print('Loss: ', loss.item())
        return model_args

    def middle_backward_compute(self, gradient_dict):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        pass_grad_keys = []
        for key, value in gradient_dict.items():
            if self.output_tensors.get(key, None) is not None:
                if isinstance(value, list):
                    for val in value:
                        if val.device.type != self.device:
                            val = val.to(self.device)

                        output_tensor = self.output_tensors[key]
                        if len(self.output_tensors) > 1 or len(value) > 1:
                            output_tensor.backward(val, retain_graph=True)
                        else:
                            output_tensor.backward(val)

                    del self.output_tensors[key]

                else:
                    if value.device.type != self.device:
                        value = value.to(self.device)

                    output_tensor = self.output_tensors[key]
                    
                    if len(self.output_tensors) > 1:
                        output_tensor.backward(value, retain_graph=True)
                    else:
                        output_tensor.backward(value)

                    del self.output_tensors[key]
            else:
                pass_grad_keys.append(key)

        load_grads_into_optimizer(self.model, self.optimizer)
        self.optimizer.step()
        load_optim_weights_into_model(self.model, self.optimizer)

        return pass_grad_keys

    def root_forward_compute(self, tensors=None):
        if not self.model.training:
            self.model.train()

        output = self.model(tensors)
        return output

    def root_no_grad_forward_compute(self, tensors=None):
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensors)

        return output

    def create_model_args(self, data, forward_pass_id=None, node_type=None):
        if node_type != NodeTypes.LEAF:
            model_args = []
            self.input_tensors[forward_pass_id] = {}
            for arg_pos, arg_metadata in self.input_template.items():
                for k, v in arg_metadata.items():

                    if isinstance(v, str) or isinstance(v, int):
                        if isinstance(v, int):
                            arg_pos = v
                        else:
                            arg_pos = 0
                        
                        if self.submod_file in data[k][arg_pos]['target']:
                            tensor_id = data[k][arg_pos]['tensor_id']

                            if data[k][arg_pos]['data'].device.type != self.device:
                                data[k][arg_pos]['data'] = data[k][arg_pos]['data'].to(self.device)

                            model_args.append(data[k][arg_pos]['data'])
                            if node_type != NodeTypes.LEAF:
                                self.input_tensors[forward_pass_id][tensor_id] = data[k][arg_pos]['data']

                            data[k][arg_pos]['target'].remove(self.submod_file)

                            if len(data[k][arg_pos]['target']) == 0:
                                del data[k][arg_pos]
                            
                            if len(data[k]) == 0:
                                del data[k]

                    elif self.submod_file in data[k][v]['target']:
                        tensor_id = data[k][v]['tensor_id']
                        if 'submod' in k or 'model_inputs' in k:                                    
                            if isinstance(v, int):    
                                
                                if data[k][v]['data'].device.type != self.device:
                                    data[k][v]['data'] = data[k][v]['data'].to(self.device)

                                model_args.append(data[k][v]['data'])
                                if node_type != NodeTypes.LEAF:
                                    if k != 'model_inputs':
                                        self.input_tensors[forward_pass_id][tensor_id] = data[k][v]['data']                                 
                            
                        data[k][v]['target'].remove(self.submod_file)

                        if len(data[k][v]['target']) == 0:
                            del data[k][v]
                        
                        if len(data[k]) == 0:
                            del data[k]
            
        else:
            model_args = {}
            for arg_pos, arg_metadata in self.input_template.items():
                for k, v in arg_metadata.items():
                    if isinstance(v, str) or isinstance(v, int):
                        if isinstance(v, int):
                            arg_pos = v
                        else:
                            arg_pos = 0
                        if self.submod_file in data[k][arg_pos]['target']:
                            tensor_id = data[k][arg_pos]['tensor_id']

                            if data[k][arg_pos]['data'].device.type != self.device:
                                data[k][arg_pos]['data'] = data[k][arg_pos]['data'].to(self.device)                            

                            model_args[tensor_id] = data[k][arg_pos]['data']
                            if node_type != NodeTypes.LEAF:
                                self.input_tensors[forward_pass_id][tensor_id] = data[k][arg_pos]['data']

                            data[k][arg_pos]['target'].remove(self.submod_file)

                            if len(data[k][arg_pos]['target']) == 0:
                                del data[k][arg_pos]
                            
                            if len(data[k]) == 0:
                                del data[k]                            
                    
                    elif self.submod_file in data[k][v]['target']:
                        tensor_id = data[k][v]['tensor_id']
                        if 'submod' in k or 'model_inputs' in k:                                    
                            if isinstance(v, int):
                                if data[k][v]['data'].device.type != self.device:
                                    data[k][v]['data'] = data[k][v]['data'].to(self.device)                                      
                                model_args[tensor_id] = data[k][v]['data']    
                            
                        data[k][v]['target'].remove(self.submod_file)

                        if len(data[k][v]['target']) == 0:
                            del data[k][v]
                        
                        if len(data[k]) == 0:
                            del data[k]
        return model_args

    def create_no_grad_model_args(self, data):
        model_args = []
        for arg_pos, arg_metadata in self.input_template.items():
            for k, v in arg_metadata.items():                 
                if isinstance(v, str) or isinstance(v, int):
                    if isinstance(v, int):
                        arg_pos = v
                    else:
                        arg_pos = 0
                    if self.submod_file in data[k][arg_pos]['target']:

                        if data[k][arg_pos]['data'].device.type != self.device:
                            data[k][arg_pos]['data'] = data[k][arg_pos]['data'].to(self.device) 
                        
                        model_args.append(data[k][arg_pos]['data'])

                        data[k][arg_pos]['target'].remove(self.submod_file)

                        if len(data[k][arg_pos]['target']) == 0:
                            del data[k][arg_pos]
                        
                        if len(data[k]) == 0:
                            del data[k]
                    
                       
                elif self.submod_file in data[k][v]['target']:
                    if 'submod' in k or 'model_inputs' in k:                                    
                        if isinstance(v, int):
                            if data[k][v]['data'].device.type != self.device:
                                data[k][v]['data'] = data[k][v]['data'].to(self.device)                                  
                            model_args.append(data[k][v]['data'])       
                                    
                    data[k][v]['target'].remove(self.submod_file)

                    if len(data[k][v]['target']) == 0:
                        del data[k][v]
                    
                    if len(data[k]) == 0:
                        del data[k]
            
        return model_args