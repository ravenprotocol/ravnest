from .strings import *
from .utils import *
import torch
import pickle
import multiprocessing
import threading

mp = multiprocessing.get_context('spawn')

class Compute():
    def __init__(self, name=None, model=None, optimizer=None, device=None, usable_gpu_memory=0.75, input_tensors=None, output_tensors=None, submod_file=None, criterion=None, input_template=None):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.usable_gpu_memory = usable_gpu_memory
        self.input_tensors = input_tensors
        self.submod_file = submod_file
        self.criterion = criterion
        self.output_tensors = output_tensors
        self.input_template = input_template
        self.fpid_to_intermediates = {}
        self.manager = mp.Manager()
        self.preload_intermediates_buffer = self.manager.dict()
        self.forward_pass_id = 0
        self.fpid_to_process_backward = 0
        self.tid = 0
        self.preload_lock = mp.Lock()
        self.current_preload_proc = None
    
    def middle_forward_compute(self, data, forward_pass_id):
        self.forward_pass_id = forward_pass_id
        model_args = self.create_model_args(data, forward_pass_id=forward_pass_id, node_type = NodeTypes.MID)
        
        if not self.model.training:
            self.model.train()
                
        if get_used_gpu_memory() >= self.usable_gpu_memory:
            with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
                output = self.model(*model_args)
        else:
            output = self.model(*model_args)
        return output
    
    def middle_no_grad_forward_compute(self, data):
        model_args = self.create_no_grad_model_args(data)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(*model_args)

        return output

    def leaf_find_loss(self, data, targets):
        model_args = self.create_model_args(data, node_type=NodeTypes.LEAF)
        
        if not self.model.training:
            self.model.train()

        outputs = self.model(*model_args.values())
        
        loss = self.criterion(outputs, targets)
        
        print('Before backward GPU: ')
        check_gpu_usage()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print('After backward GPU: ')
        check_gpu_usage()

        print('Loss: ', round(loss.item(), 4))
        f = open("losses.txt", "a")
        f.write(str(round(loss.item(), 4)) + '\n')
        f.close() 
        return model_args

    def middle_backward_compute(self, gradient_dict, forward_pass_id):
        self.fpid_to_process_backward = forward_pass_id

        if self.current_preload_proc is not None and self.current_preload_proc.is_alive():
            self.current_preload_proc.join()
        
        if self.fpid_to_intermediates.get(forward_pass_id+1, None) is not None:
            self.current_preload_proc = mp.Process(target=self.trigger_parallel_preload_jobs, args=(forward_pass_id+1, self.fpid_to_intermediates[forward_pass_id+1],))
            self.current_preload_proc.start()

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

    def root_forward_compute(self, tensors=None, forward_pass_id=None):
        self.forward_pass_id = forward_pass_id
        if not self.model.training:
            self.model.train()
        
        if get_used_gpu_memory() >= self.usable_gpu_memory:
            with torch.autograd.graph.saved_tensors_hooks(self.pack_hook, self.unpack_hook):
                output = self.model(tensors)
        else:
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
                                data[k][arg_pos]['data'].retain_grad()

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
                                    data[k][v]['data'].retain_grad()

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
                                data[k][arg_pos]['data'].retain_grad()                        

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
                                    data[k][v]['data'].retain_grad()                            
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

    def pack_hook(self, tensor):
        temp_file_name = '{}_aux/{}.pt'.format(self.tid, self.name)
        temp_file = SelfDeletingTempFile(name=temp_file_name)
        torch.save(tensor, temp_file.name, pickle_module=pickle, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        self.tid += 1
        if self.fpid_to_intermediates.get(self.forward_pass_id, None) is not None:
            self.fpid_to_intermediates[self.forward_pass_id].append(temp_file.name)
        else:
            self.fpid_to_intermediates[self.forward_pass_id] = [temp_file.name]
        
        return temp_file

    def unpack_hook(self, file):
        self.preload_lock.acquire()
        if self.preload_intermediates_buffer.get(self.fpid_to_process_backward, None) is not None:
            tensor = self.preload_intermediates_buffer[self.fpid_to_process_backward][file.name]
            temp = self.preload_intermediates_buffer[self.fpid_to_process_backward]
            del temp[file.name]
            self.preload_intermediates_buffer[self.fpid_to_process_backward] = temp
            self.preload_lock.release()
            return tensor
        self.preload_lock.release()
        return torch.load(file.name)

    def trigger_parallel_preload_jobs(self, next_fpid, intermediate_tensors):
        k, m = divmod(len(intermediate_tensors), 3)
        chunked_list = (intermediate_tensors[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(3))
        preload_threads = []
        for chunk in chunked_list:
            t = threading.Thread(target=self.populate_preload_buffer, args=(next_fpid, chunk))
            preload_threads.append(t)
            t.start()
        
        for thread in preload_threads:
            thread.join()

    
    def populate_preload_buffer(self, next_fpid, intermediate_tensors):
        fpid_preload_buffer = {}
        for file_name in intermediate_tensors:
            fpid_preload_buffer[file_name] = torch.load(file_name)

        self.preload_lock.acquire()
        self.preload_intermediates_buffer[next_fpid] = fpid_preload_buffer
        self.preload_lock.release()

    
    def __getstate__(self):
        return dict(
            preload_lock = self.preload_lock,
            preload_intermediates_buffer = self.preload_intermediates_buffer
        )

    def __setstate__(self, state):
        self.preload_lock = state['preload_lock']
        self.preload_intermediates_buffer = state['preload_intermediates_buffer']