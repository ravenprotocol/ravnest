import copy
import torch
import threading
import time
from .utils import *
from .strings import *
from .pipeline_split.modeling import get_pipeline_stage_submod

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.jit.set_fusion_strategy([('STATIC',0), ('DYNAMIC', 0)])

class Compute():
    def __init__(self, model = None, optimizer = None, compression = False,
                latest_weights_buffer=None, latest_weights_lock=None,
                input_tensors = None, tensor_id = None, rank=None,
                layer_start_idx=None, layer_end_idx=None,
                output_template = None, input_template = None,
                node_type=None, backend='grpc', recompute=False,
                submod_file = None, loss_filename = None, device = None):
        self.model = model
        self.optimizer = optimizer
        self.compression = compression
        self.input_tensors = input_tensors
        self.current_version = 0
        self.fpid_to_version = {}
        self.version_to_fpid = {}
        self.version_to_param = {}
        self.latest_weights_lock = latest_weights_lock
        self.latest_weights_buffer = latest_weights_buffer
        self.fpid_to_rng = {}
        self.submod_file = submod_file
        self.loss_filename = loss_filename
        self.output_tensors = {}
        self.tensor_id = tensor_id
        self.output_template = output_template
        self.input_template = input_template
        self.node_type = node_type
        self.backend = backend
        self.recompute = recompute
        self.device = device
        self.recompute_thread = None
        self.file_loss = 0
        self.version_update_lock = threading.Lock()

        self.pipeline_stage = get_pipeline_stage_submod(self.model, rank, self.node_type, layer_start_idx, layer_end_idx)
        self.update_model_version()

    def update_model_version(self):
        if self.recompute:
            self.version_to_param[self.current_version] = self.get_params_clone()

    def root_forward_compute(self, tensors, forward_pass_id, **kwargs):
        if self.recompute_thread is not None:
            if self.recompute_thread.is_alive():
                self.recompute_thread.join()

        if len(kwargs) > 0:
            if tensors is not None:
                self.input_tensors[forward_pass_id] = {'tensors': tensors, 'kwargs':kwargs}
            else:
                self.input_tensors[forward_pass_id] = {'kwargs':kwargs}

        else:
            self.input_tensors[forward_pass_id] = tensors

        if tensors is not None:
            output = self.pipeline_stage.forward(tensors, **kwargs)
        else:
            output = self.pipeline_stage.forward(**kwargs)
        
        self.output_tensors[forward_pass_id] = output

        if self.version_to_fpid.get(self.current_version, None) is not None:
            self.version_to_fpid[self.current_version].append(forward_pass_id)
        else:
            self.version_to_fpid[self.current_version] = [forward_pass_id]
        
        self.fpid_to_version[forward_pass_id] = self.current_version
        return output

    def middle_forward_compute(self, data, forward_pass_id):
        if self.backend == 'grpc':
            model_args = self.create_model_args(data, forward_pass_id=forward_pass_id, node_type = NodeTypes.STEM)
        else:
            model_args = data
            for model_arg in model_args:
                model_arg.requires_grad_()
            self.input_tensors[forward_pass_id] = data

        if isinstance(model_args, torch.Tensor):
            output = self.model(model_args)
        else:    
            output = self.model(*model_args)
        
        self.output_tensors[forward_pass_id] = output

        if self.version_to_fpid.get(self.current_version, None) is not None:
            self.version_to_fpid[self.current_version].append(forward_pass_id)
        else:
            self.version_to_fpid[self.current_version] = [forward_pass_id]
        
        self.fpid_to_version[forward_pass_id] = self.current_version

        return output

    def num_grad_enabled_output_tensors(self):
        num_grad_enabled = 0
        for k, v in self.output_tensors.items():
            if v.grad_fn is not None:
                num_grad_enabled += 1
        return num_grad_enabled

    def middle_backward_compute(self, gradient_data, forward_pass_id):

        pass_grad_keys = []
        leaf_output_tensors = []
        backward_grads = []

        if self.backend == 'grpc':
            for key, value in gradient_data.items():
                if self.output_tensors.get(key, None) is not None:
                    if self.output_tensors[key].grad_fn is not None:
                        original_dtype = value['dtype']
                        value = value['data']
                        if self.compression:
                            value = extract_tensor_from_compression_float16(value, original_dtype)
                        
                        if value.device.type != self.device:
                            value = value.to(self.device)

                        output_tensor = self.output_tensors[key]
                        leaf_output_tensors.append(output_tensor)
                        backward_grads.append(value)

                        del self.output_tensors[key]
                    else:
                        del self.output_tensors[key]
                else:
                    pass_grad_keys.append(key)
        else:
            leaf_output_tensors = self.output_tensors[forward_pass_id]
            backward_grads = gradient_data
        torch.autograd.backward(leaf_output_tensors, backward_grads)
        del self.output_tensors[forward_pass_id]

        recompute_version = self.fpid_to_version[forward_pass_id]
        self.version_to_fpid[recompute_version].remove(forward_pass_id)
        if len(self.version_to_fpid[recompute_version]) == 0:
            del self.version_to_fpid[recompute_version]
        
        return pass_grad_keys

    def recompute_forward(self, forward_pass_id):
        recompute_version = self.fpid_to_version[forward_pass_id]
        del self.fpid_to_version[forward_pass_id]

        load_state_dict_conserve_versions(self.model, self.version_to_param[recompute_version])
        if not self.model.training:
            self.model.train()

        cpu_rng, gpu_rng = self.fpid_to_rng[forward_pass_id]
        del self.fpid_to_rng[forward_pass_id]
        devices = []

        if self.device.type == 'cuda':
            devices.append(self.device)

        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:
                torch.cuda.set_rng_state(gpu_rng, self.device)

            if self.node_type == NodeTypes.ROOT:
                if isinstance(self.input_tensors[forward_pass_id], dict):
                    if 'tensors' in self.input_tensors[forward_pass_id].keys():
                        output = self.model(self.input_tensors[forward_pass_id]['tensors'], **self.input_tensors[forward_pass_id]['kwargs'])
                    else:
                        output = self.model(**self.input_tensors[forward_pass_id]['kwargs'])
                else:
                    output = self.model(self.input_tensors[forward_pass_id])
            else:
                if self.backend == 'grpc':
                    output = self.model(*self.get_model_args(forward_pass_id))
                else:
                    if isinstance(self.input_tensors[forward_pass_id], torch.Tensor):
                        output = self.model(self.input_tensors[forward_pass_id])
                    else:
                        output = self.model(*self.input_tensors[forward_pass_id])
        
        if self.backend == 'grpc':
            for k, v in self.output_template.items():
                if isinstance(output, tuple):
                    out = output[k]
                else:
                    out = output
                
                self.output_tensors[self.tensor_id] = out
                self.tensor_id = str(int(self.tensor_id.split('_')[0]) + 1) + '_{}'.format(self.submod_file)
        else:
            self.output_tensors = output
        
        self.version_update_lock.acquire(blocking=True)
        load_state_dict_conserve_versions(self.model, self.version_to_param[self.current_version])
        self.version_update_lock.release()

        self.version_to_fpid[recompute_version].remove(forward_pass_id)
        if len(self.version_to_fpid[recompute_version]) == 0:
            del self.version_to_param[recompute_version]
            del self.version_to_fpid[recompute_version]

    def leaf_backward(self, loss):
        loss.backward()
    
    def leaf_forward(self, data):
        if self.backend == 'grpc':
            model_args = self.create_model_args(data, node_type=NodeTypes.LEAF)
            outputs = self.model(*model_args.values())
        else:
            model_args = data
            if isinstance(model_args, torch.Tensor):
                model_args.requires_grad_()
                outputs = self.model(model_args)
            else:
                for model_arg in model_args:
                    model_arg.requires_grad_()
                outputs = self.model(*model_args)

        return model_args, outputs

    def root_no_grad_forward_compute(self, tensors=None, **kwargs):
        with torch.no_grad():
            if tensors is not None:
                output = self.pipeline_stage.forward(tensors, **kwargs)
            else:
                output = self.pipeline_stage.forward(**kwargs)
        return output

    def middle_no_grad_forward_compute(self, **kwargs):
        with torch.no_grad():
            output = self.pipeline_stage.forward(**kwargs)
        return output
    
    def leaf_no_grad_forward(self, **kwargs):
        with torch.no_grad():
            output = self.pipeline_stage.forward(**kwargs)
        return output

    def optimizer_step(self):
        if self.node_type == NodeTypes.LEAF:
            self.optimizer.step()
        else:
            load_grads_into_optimizer(self.model, self.optimizer)
            self.optimizer.step()
            load_optim_weights_into_model(self.model, self.optimizer)

            self.version_update_lock.acquire(blocking=True)
            if self.version_to_fpid.get(self.current_version, None) is None:
                if self.current_version in self.version_to_param:
                    del self.version_to_param[self.current_version]
            self.current_version += 1
            self.update_model_version()
            self.version_update_lock.release()

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

                            model_arg = data[k][arg_pos]['data']
                            original_dtype = data[k][arg_pos]['dtype']
                            if self.compression:
                                model_arg = extract_tensor_from_compression_float16(model_arg, original_dtype)

                            if model_arg.device.type != self.device:
                                model_arg = model_arg.to(self.device)
                            
                            model_arg.requires_grad_()
                            model_args.append(model_arg)
                            self.input_tensors[forward_pass_id][tensor_id] = model_arg

                            data[k][arg_pos]['target'].remove(self.submod_file)

                            if len(data[k][arg_pos]['target']) == 0:
                                del data[k][arg_pos]
                            
                            if len(data[k]) == 0:
                                del data[k]

                    elif self.submod_file in data[k][v]['target']:
                        tensor_id = data[k][v]['tensor_id']
                        if 'submod' in k or 'model_inputs' in k:                                    
                            if isinstance(v, int):    
                                model_arg = data[k][v]['data']

                                original_dtype = data[k][arg_pos]['dtype']
                                if self.compression:
                                    model_arg = extract_tensor_from_compression_float16(model_arg, original_dtype)

                                if model_arg.device.type != self.device:
                                    model_arg = model_arg.to(self.device)
                                
                                model_arg.requires_grad_()
                                model_args.append(model_arg)
                                self.input_tensors[forward_pass_id][tensor_id] = model_arg                              
                            
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

                            model_arg = data[k][arg_pos]['data']

                            original_dtype = data[k][arg_pos]['dtype']
                            if self.compression:
                                model_arg = extract_tensor_from_compression_float16(model_arg, original_dtype)
                            
                            if model_arg.device.type != self.device:
                                model_arg = model_arg.to(self.device)    
                            
                            model_arg.requires_grad_()                     
                            model_args[tensor_id] = model_arg
                            if node_type != NodeTypes.LEAF:
                                self.input_tensors[forward_pass_id][tensor_id] = model_arg

                            data[k][arg_pos]['target'].remove(self.submod_file)

                            if len(data[k][arg_pos]['target']) == 0:
                                del data[k][arg_pos]
                            
                            if len(data[k]) == 0:
                                del data[k]                            
                    
                    elif self.submod_file in data[k][v]['target']:
                        tensor_id = data[k][v]['tensor_id']
                        if 'submod' in k or 'model_inputs' in k:                                    
                            if isinstance(v, int):

                                model_arg = data[k][v]['data']

                                original_dtype = data[k][arg_pos]['dtype']
                                if self.compression:
                                    model_arg = extract_tensor_from_compression_float16(model_arg, original_dtype)
                                
                                if model_arg.device.type != self.device:
                                    model_arg = model_arg.to(self.device) 
                            
                                model_arg.requires_grad_()                            
                                model_args[tensor_id] = model_arg 
                            
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

    def get_model_args(self, forward_pass_id):
        model_args = []
        for tid, input in self.input_tensors[forward_pass_id].items():
            model_args.append(input)

        return tuple(model_args)

    @torch.no_grad()
    def get_params_clone(self):

        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].clone()
        return state_dict
