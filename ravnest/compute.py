from .strings import *
from .utils import *
import torch
import pickle
# import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

mp = torch.multiprocessing.get_context('spawn')

class Compute():
    def __init__(self, name=None, model=None, 
                 optimizer=None, device=None, 
                 node_type = None,
                 output_template = None,
                 usable_gpu_memory=0.75, 
                 input_tensors=None, output_tensors=None, 
                 fpid_to_tensor_ids=None, submod_file=None, 
                 criterion=None, input_template=None):
    
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.node_type = node_type
        self.usable_gpu_memory = usable_gpu_memory
        self.input_tensors = input_tensors
        self.submod_file = submod_file
        self.criterion = criterion
        self.output_tensors = output_tensors
        self.input_template = input_template
        self.output_template = output_template
        self.active_version = None

        self.recompute_thread = None

        self.fpid_to_tensor_ids = fpid_to_tensor_ids
        self.fpid_to_version = {}
        self.version_to_fpid = {}
        self.version_to_param = {}
        self.current_version = 0
        self.fpid_to_rng = {}
    
    def middle_forward_compute(self, data, forward_pass_id):

        if self.recompute_thread is not None:
            if self.recompute_thread.is_alive():
                self.recompute_thread.join()

        print('Middle Forward fpid: ',forward_pass_id)
        model_args = self.create_model_args(data, forward_pass_id=forward_pass_id, node_type = NodeTypes.MID)

        rng_state_cpu = torch.get_rng_state()
        rng_state_gpu = None
        if self.device.type == 'cuda':
            rng_state_gpu = torch.cuda.get_rng_state(self.device)
        
        self.fpid_to_rng[forward_pass_id] = (rng_state_cpu, rng_state_gpu)

        # print('Active version in forward: ', self.active_version)
        with torch.no_grad():
            output = self.model(*model_args)

        if self.version_to_fpid.get(self.current_version, None) is not None:
            self.version_to_fpid[self.current_version].append(forward_pass_id)
        else:
            self.version_to_fpid[self.current_version] = [forward_pass_id]
        
        self.fpid_to_version[forward_pass_id] = self.current_version
        if self.current_version not in self.version_to_param:
            # print('Adding current version in forward: ', self.current_version)
            self.version_to_param[self.current_version] = self.model.state_dict()

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
        
        # print('Before backward GPU: ')
        # check_gpu_usage()
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print('After backward GPU: ')
        # check_gpu_usage()

        print('Loss: ', round(loss.item(), 4))
        f = open("losses.txt", "a")
        f.write(str(round(loss.item(), 4)) + '\n')
        f.close() 
        return model_args

    def middle_backward_compute(self, gradient_dict, forward_pass_id):
        print('\nMiddle Backward Compute Started', forward_pass_id)

        if self.fpid_to_version.get(forward_pass_id, None) is not None:
            print('Blocking recompute for: ', forward_pass_id)
            self.recompute_forward(forward_pass_id)

        # for param in self.model.parameters():
        #     print('Param version before backward: ', param._version)
        #     break

        if self.recompute_thread is not None:
            if self.recompute_thread.is_alive():
                self.recompute_thread.join()

        # print('Active version in backward: ', self.active_version)
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

        # for param in self.model.parameters():
        #     print('Param version before step: ', param._version)
        #     break

        load_grads_into_optimizer(self.model, self.optimizer)
        self.optimizer.step()
        load_optim_weights_into_model(self.model, self.optimizer)

        # for param in self.model.parameters():
        #     print('Param version after step: ', param._version)
        #     break

        self.current_version += 1
        self.active_version = self.current_version
        self.version_to_param[self.current_version] = self.model.state_dict()

        if self.fpid_to_version.get(forward_pass_id+1, None) is not None:
            self.recompute_thread = threading.Thread(target=self.recompute_forward, args=(forward_pass_id+1,))
            self.recompute_thread.start()

        return pass_grad_keys

    def root_forward_compute(self, tensors=None, forward_pass_id=None):
        if self.recompute_thread is not None:
            if self.recompute_thread.is_alive():
                self.recompute_thread.join()

        rng_state_cpu = torch.get_rng_state()
        rng_state_gpu = None
        if self.device.type == 'cuda':
            rng_state_gpu = torch.cuda.get_rng_state(self.device)
        
        # if not self.model.training:
        #     self.model.train()
        
        self.fpid_to_rng[forward_pass_id] = (rng_state_cpu, rng_state_gpu)
        self.input_tensors[forward_pass_id] = tensors
        with torch.no_grad():
            output = self.model(tensors)

        if self.version_to_fpid.get(self.current_version, None) is not None:
            self.version_to_fpid[self.current_version].append(forward_pass_id)
        else:
            self.version_to_fpid[self.current_version] = [forward_pass_id]

        self.fpid_to_version[forward_pass_id] = self.current_version

        if self.current_version not in self.version_to_param:
            # print('Adding current version in root forward: ', self.current_version)
            self.version_to_param[self.current_version] = self.model.state_dict()

        return output

    def root_no_grad_forward_compute(self, tensors=None):
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensors)

        return output
    
    def recompute_forward(self, forward_pass_id):
        print('Recompute Starting for: ', forward_pass_id, self.current_version)
        recompute_version = self.fpid_to_version[forward_pass_id]
        # print('Recompute version: ', recompute_version)
        del self.fpid_to_version[forward_pass_id]

        # for param in self.model.parameters():
        #     print('Param version in recompute, before load old: ', param._version)
        #     break

        if self.current_version != recompute_version:
            # self.model.load_state_dict(self.version_to_param[recompute_version])
            load_state_dict_conserve_versions(self.model, self.version_to_param[recompute_version])
            self.active_version = recompute_version
        # print('Active version in recompute: ', self.active_version)
        # for param in self.model.parameters():
        #     print('Param version in recompute, after load old: ', param._version)
        #     break

        cpu_rng, gpu_rng = self.fpid_to_rng[forward_pass_id]
        del self.fpid_to_rng[forward_pass_id]
        devices = []

        if self.device.type == 'cuda':
            devices.append(self.device)
        
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:
                torch.cuda.set_rng_state(gpu_rng)

            if self.node_type == NodeTypes.ROOT:
                output = self.model(self.input_tensors[forward_pass_id])
            else:
                output = self.model(*self.get_model_args(forward_pass_id))

        for k,v in self.output_template.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output

            self.output_tensors[self.fpid_to_tensor_ids[forward_pass_id][k]] = out

        del self.fpid_to_tensor_ids[forward_pass_id]

        self.version_to_fpid[recompute_version].remove(forward_pass_id)
        if len(self.version_to_fpid[recompute_version]) == 0:
            del self.version_to_fpid[recompute_version]
            del self.version_to_param[recompute_version]

        # for param in self.model.parameters():
        #     print('Param version after recompute and before reload: ', param._version)
        #     break

        if self.current_version != recompute_version:
            # print('Current version before load state: ', forward_pass_id, self.current_version)
            # with torch.no_grad():
            #     self.model.load_state_dict(self.version_to_param[self.current_version])
            load_state_dict_conserve_versions(self.model, self.version_to_param[self.current_version])
            self.active_version = self.current_version

        # print('Active version in recompute end: ', self.active_version)
        # for param in self.model.parameters():
        #     print('Param version after recompute and after reload: ', param._version)
        #     break
        print('Recompute Finished for: ', forward_pass_id)


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
                                data[k][arg_pos]['data'].requires_grad_()

                            model_args.append(data[k][arg_pos]['data'])
                            # if node_type != NodeTypes.LEAF:
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
                                    data[k][v]['data'].requires_grad_()

                                model_args.append(data[k][v]['data'])
                                # if node_type != NodeTypes.LEAF:
                                # if k != 'model_inputs':
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
                                data[k][arg_pos]['data'].requires_grad_()                     

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
                                    data[k][v]['data'].requires_grad_()                            
                                model_args[tensor_id] = data[k][v]['data']    
                            
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