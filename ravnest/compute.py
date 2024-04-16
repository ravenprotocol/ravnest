import copy
import threading
from .utils import *
from .strings import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.jit.set_fusion_strategy([('STATIC',0), ('DYNAMIC', 0)])

class Compute():
    def __init__(self, model = None, optimizer = None, 
                criterion = None,
                input_tensors = None, tensor_id = None, 
                output_template = None, input_template = None,
                node_type=None,
                submod_file = None, device = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.input_tensors = input_tensors
        self.current_version = 0
        self.fpid_to_version = {}
        self.version_to_fpid = {}
        self.version_to_param = {}
        self.fpid_to_rng = {}
        self.submod_file = submod_file
        self.output_tensors = {}
        self.tensor_id = tensor_id
        self.output_template = output_template
        self.input_template = input_template
        self.node_type = node_type
        self.device = device
        self.recompute_thread = None
        # self.recompute_stream = torch.cuda.Stream(self.device)
        self.version_to_param[self.current_version] = self.get_params_clone()

    def root_forward_compute(self, tensors, forward_pass_id):
        if self.recompute_thread is not None:
            if self.recompute_thread.is_alive():
                self.recompute_thread.join()

        # torch.cuda.synchronize()

        if not self.model.training:
            self.model.train()

        rng_state_cpu = torch.get_rng_state()
        rng_state_gpu = None
        if self.device.type == 'cuda':
            rng_state_gpu = torch.cuda.get_rng_state(self.device)
        
        self.fpid_to_rng[forward_pass_id] = (rng_state_cpu, rng_state_gpu)

        self.input_tensors[forward_pass_id] = tensors

        with torch.no_grad():
            output = self.model(tensors)

        if self.version_to_fpid.get(self.current_version, None) is not None:
            self.version_to_fpid[self.current_version].append(forward_pass_id)
        else:
            self.version_to_fpid[self.current_version] = [forward_pass_id]
        
        self.fpid_to_version[forward_pass_id] = self.current_version
        print('Forward done for: ', forward_pass_id)
        return output

    def middle_forward_compute(self, data, forward_pass_id):
        if self.recompute_thread is not None:
            if self.recompute_thread.is_alive():
                self.recompute_thread.join()

        # torch.cuda.synchronize()

        if not self.model.training:
            self.model.train()

        print('Middle Forward fpid: ',forward_pass_id)
        model_args = self.create_model_args(data, forward_pass_id=forward_pass_id, node_type = NodeTypes.MID)

        rng_state_cpu = torch.get_rng_state()
        rng_state_gpu = None
        if self.device.type == 'cuda':
            rng_state_gpu = torch.cuda.get_rng_state(self.device)
        
        self.fpid_to_rng[forward_pass_id] = (rng_state_cpu, rng_state_gpu)

        with torch.no_grad():
            output = self.model(*model_args)

        if self.version_to_fpid.get(self.current_version, None) is not None:
            self.version_to_fpid[self.current_version].append(forward_pass_id)
        else:
            self.version_to_fpid[self.current_version] = [forward_pass_id]
        
        self.fpid_to_version[forward_pass_id] = self.current_version

        return output

    def middle_backward_compute(self, gradient_dict, forward_pass_id):
        # self.recompute_forward(forward_pass_id)
        if self.recompute_thread is not None:
            if self.recompute_thread.is_alive():
                self.recompute_thread.join()

        # torch.cuda.synchronize()

        if self.fpid_to_version.get(forward_pass_id, None) is not None:
            print('Blocking recompute for: ', forward_pass_id)
            self.recompute_forward(forward_pass_id)

        self.model.zero_grad()
        self.optimizer.zero_grad()
        pass_grad_keys = []
        # print('Gradient dict: ', gradient_dict.keys())
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

        if self.version_to_fpid.get(self.current_version, None) is None:
            if self.current_version in self.version_to_param:
                del self.version_to_param[self.current_version]

        self.current_version += 1
        self.version_to_param[self.current_version] = self.get_params_clone()
        print('Len of dictionaries: ', len(self.fpid_to_version), len(self.version_to_fpid), len(self.version_to_param), len(self.output_tensors))
        
        if self.fpid_to_version.get(forward_pass_id+1, None) is not None:
            self.recompute_thread = threading.Thread(target=self.recompute_forward, args=(forward_pass_id+1,))
            self.recompute_thread.start()

        # if self.fpid_to_version.get(forward_pass_id+1, None) is not None:
        #     with torch.cuda.stream(self.recompute_stream): #self.recompute_stream
        #         self.recompute_forward(forward_pass_id+1)
        
        return pass_grad_keys

    def recompute_forward(self, forward_pass_id):
        recompute_version = self.fpid_to_version[forward_pass_id]
        del self.fpid_to_version[forward_pass_id]

        load_state_dict_conserve_versions(self.model, self.version_to_param[recompute_version])
        # inputs = self.input_tensors[forward_pass_id]

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
                output = self.model(self.input_tensors[forward_pass_id])
            else:
                output = self.model(*self.get_model_args(forward_pass_id))

        for k, v in self.output_template.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output
            # print('Recompute tensor_id: ', self.tensor_id)
            self.output_tensors[self.tensor_id] = out
            self.tensor_id = str(int(self.tensor_id.split('_')[0]) + 1) + '_{}'.format(self.submod_file)

        load_state_dict_conserve_versions(self.model, self.version_to_param[self.current_version])

        self.version_to_fpid[recompute_version].remove(forward_pass_id)
        if len(self.version_to_fpid[recompute_version]) == 0:
            print('Deleting in recompute: Recompute version ', recompute_version)
            del self.version_to_param[recompute_version]
            del self.version_to_fpid[recompute_version]
        print('Recompute done for: ', forward_pass_id)

    def leaf_find_loss(self, data, targets):
        model_args = self.create_model_args(data, node_type=NodeTypes.LEAF)
        
        if not self.model.training:
            self.model.train()

        outputs = self.model(*model_args.values())
        # print(outputs, targets)
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

    def root_no_grad_forward_compute(self, tensors=None):
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensors)

        return output

    def middle_no_grad_forward_compute(self, data):
        model_args = self.create_no_grad_model_args(data)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(*model_args)

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
                                data[k][arg_pos]['data'] = data[k][arg_pos]['data'].detach().clone().to(self.device)
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
                                    data[k][v]['data'] = data[k][v]['data'].detach().clone().to(self.device)
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
                                data[k][arg_pos]['data'] = data[k][arg_pos]['data'].detach().clone().to(self.device)    
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
                                    data[k][v]['data'] = data[k][v]['data'].detach().clone().to(self.device) 
                                    data[k][v]['data'].requires_grad_()                            
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

    def get_model_args(self, forward_pass_id):
        model_args = []
        for tid, input in self.input_tensors[forward_pass_id].items():
            model_args.append(input)

        return tuple(model_args)

    @torch.no_grad()
    def get_params_clone(self):
        # state_dict = self.model.state_dict()
        # for key in state_dict:
        #     state_dict[key] = torch.empty_like(state_dict[key]).copy_(state_dict[key])
        # return state_dict

        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].clone()
        return state_dict

def compare_models(d1, d2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(d1.items(), d2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
                break
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')