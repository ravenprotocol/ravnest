from concurrent import futures
import asyncio
import grpc
import threading
import multiprocessing as mp
from threading import Thread
import numpy as np
import itertools
import psutil
import pickle
import time
from ravnest.utils import *
from ravnest.strings import *
from ravnest.endpoints import GrpcService

from protos.server_pb2_grpc import add_CommServerServicer_to_server, CommServerStub
from protos.server_pb2 import CheckBufferStatus, CheckReduceIteration, CheckGatherIteration

class Node():
    def __init__(self, name=None, model=None, optimizer=None, criterion=None, 
                 labels=None, test_labels=None, device = 'cpu', **kwargs):
        self.manager = mp.Manager()
        self.forward_lock = mp.Lock()
        self.backward_lock = mp.Lock()
        self.reduce_lock = mp.Lock()
        self.gather_lock = mp.Lock()

        self.local_address = '{}:{}'.format(kwargs.get('local_host', None), kwargs.get('local_port', None))
        self.name = name

        self.model = model
        self.device = device

        if not next(self.model.parameters()).is_cuda:
            self.model.to(device)

        self.load_forward_buffer = self.manager.list()
        self.load_backward_buffer = self.manager.list()
        self.reduce_ring_buffers = self.manager.dict()
        self.gather_ring_buffers = self.manager.dict()
        self.reduce_iteration = self.manager.dict()
        self.gather_iteration = self.manager.dict()

        if kwargs.get('ring_ids', None) is not None:
            self.ring_ids = kwargs.get('ring_ids', None)

            for ring_id, _ in self.ring_ids.items():
                self.reduce_iteration[ring_id] = 0
                self.gather_iteration[ring_id] = 0
            print('ring ids: ', self.ring_ids)

        self.rank = kwargs.get('rank', None)
        self.ring_size = kwargs.get('ring_size', None)
        
        self.ring_param_keys = {}
        # self.data_dict = data_dict
        # if data_dict is not None:
        # data_dict_keys = list(data_dict.keys())
        data_dict_keys = get_trainable_param_names(model=self.model)
        for i, ring in enumerate(self.ring_ids.items()):
            if i < len(self.ring_ids) - 1:
                keys = data_dict_keys[data_dict_keys.index(ring[1]):data_dict_keys.index(self.ring_ids[ring[0]+1])]
            else:
                keys = data_dict_keys[data_dict_keys.index(ring[1]):]
            
            self.ring_param_keys[ring[0]] = keys

        self.param_address_mapping = {}
        param_addresses = kwargs.get('param_addresses', None)
        for i, address_to_param in enumerate(param_addresses.items()):
            if i < len(param_addresses) - 1:
                keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):data_dict_keys.index(param_addresses[list(param_addresses.keys())[i+1]])]
            else:
                keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):]
            
            for param_name in keys:
                self.param_address_mapping[param_name] = address_to_param[0]

        # self.send_buffer = []
        self.criterion = criterion
        self.labels = labels
        self.test_labels = test_labels
        self.forward_target_host = kwargs.get('forward_target_host', None)
        self.forward_target_port = kwargs.get('forward_target_port', None)
        self.backward_target_host = kwargs.get('backward_target_host', None)
        self.backward_target_port = kwargs.get('backward_target_port', None)

        self.output_tensors = {}
        self.input_tensors = {}
        self.n_backwards = 0
        self.forward_pass_id = 0

        self.reduce_threshold = 8

        self.submod_file = kwargs.get('submod_file', None)
        self.node_status = NodeStatus.IDLE
        self.tensor_id = '0_{}'.format(self.submod_file)#0

        self.averaged_params_buffer = {}
        self.average_no = 0

        if kwargs.get('submod_file', None) is not None:
            with open('{}{}_input.pkl'.format(kwargs.get('template_path', None), kwargs.get('submod_file', None)), 'rb') as fout:
                self.input_template = pickle.load(fout)
            with open('{}{}_output.pkl'.format(kwargs.get('template_path', None), kwargs.get('submod_file', None)), 'rb') as fout:
                self.output_template = pickle.load(fout)
            print(self.input_template)
            if self.backward_target_host is None and self.backward_target_port is None:
                self.node_type = NodeTypes.ROOT
                with open('{}model_inputs.pkl'.format(kwargs.get('template_path', None)), 'rb') as fout:
                    self.model_inputs_template = pickle.load(fout)
                self.optimizer = optimizer(current_model_params_clone(self.model))
            elif self.forward_target_host is None and self.forward_target_port is None:
                self.node_type = NodeTypes.LEAF
                self.optimizer = optimizer(self.model.parameters())
            else:
                self.node_type = NodeTypes.MID
                self.optimizer = optimizer(current_model_params_clone(self.model))

    def init_server(self, load_forward_buffer=None, load_backward_buffer=None, 
                    reduce_ring_buffers = None, gather_ring_buffers = None, 
                    forward_lock=None, backward_lock=None, reduce_lock=None, gather_lock=None,
                    reduce_iteration = None, gather_iteration = None):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_CommServerServicer_to_server(GrpcService(
            load_forward_buffer=load_forward_buffer, load_backward_buffer=load_backward_buffer, 
            reduce_ring_buffers=reduce_ring_buffers, gather_ring_buffers=gather_ring_buffers,
            forward_lock=forward_lock, backward_lock=backward_lock, reduce_lock=reduce_lock, gather_lock=gather_lock,
            reduce_iteration = reduce_iteration, gather_iteration = gather_iteration), self.server)
        print('Length of forward buffer: ', len(load_backward_buffer), os.getpid())


    def grpc_server_serve(self):
        print('Listening on : ', self.local_address)
        self.server.add_insecure_port(self.local_address)
        self.server.start()
        self.server.wait_for_termination()


    def start_grpc_server(self):
        asyncio.get_event_loop().run_until_complete(self.grpc_server_serve())

    def start(self):
        print('Main process: ', os.getpid())
        serve_process = mp.Process(target=self.grpc_server_serve, daemon=True)
        serve_process.start()
        time.sleep(2)
        buffer_thread = threading.Thread(target=self.check_load_forward_buffer, daemon=True)
        buffer_thread.start()

    def check_load_forward_buffer(self):
        while True:
            # print('Backward: ', len(self.load_backward_buffer))
            # print('Forward: ', len(self.load_forward_buffer))
            send_trigger_threads = []
            if len(self.load_backward_buffer) != 0:
                self.backward_lock.acquire(block=True)
                value = self.load_backward_buffer[0]
                del self.load_backward_buffer[0]
                self.backward_lock.release()
                action = value['action']

                if action == ActionTypes.BACKWARD:
                    self.node_status = NodeStatus.BACKWARD
                    gradient_dict = value['data']
                    forward_pass_id = value['forward_pass_id']
                    
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    # print('gradient dict ', gradient_dict)
                    # print('output tensors ', self.output_tensors)
                    # print('input tensors: ', self.input_tensors)
                    pass_grad_keys = []
                    for key, value in gradient_dict.items():
                        if self.output_tensors.get(key, None) is not None:
                            if isinstance(value, list):
                                for val in value:
                                    if val.device.type != self.device:
                                        val = val.to(self.device)

                                    output_tensor = self.output_tensors[key]
                                    # print('output tensor in val list: ', output_tensor)
                                    if len(self.output_tensors) > 1 or len(value) > 1:
                                        output_tensor.backward(val, retain_graph=True)
                                    else:
                                        output_tensor.backward(val)

                                del self.output_tensors[key]

                            else:
                                if value.device.type != self.device:
                                    value = value.to(self.device)

                                output_tensor = self.output_tensors[key]
                                # print('output tensor in not val list: ', output_tensor)

                                if len(self.output_tensors) > 1:
                                    output_tensor.backward(value, retain_graph=True)
                                else:
                                    output_tensor.backward(value)

                                del self.output_tensors[key]
                        else:
                            pass_grad_keys.append(key)

                    # print('\n Param grads')
                    # for param in self.model.parameters():
                    #     if param.requires_grad:
                    #         print(param.grad)

                    load_grads_into_optimizer(self.model, self.optimizer)
                    self.optimizer.step()
                    load_optim_weights_into_model(self.model, self.optimizer)

                    if self.node_type != NodeTypes.ROOT:
                        gradients = self.create_backward_payload(forward_pass_id=forward_pass_id)
                        # self.send_buffer.append({'action':ActionTypes.BACKWARD,
                        #                         'forward_pass_id':forward_pass_id, 
                        #                         'data':gradients, 
                        #                         })
                        # print('\n Gradients before: ', gradients)
                        for pass_key in pass_grad_keys:
                            if pass_key in gradients.keys():
                                if isinstance(gradient_dict[pass_key], list):
                                    gradient_dict[pass_key].append(gradients[pass_key])
                                    gradients[pass_key] = gradient_dict[pass_key]
                                else:
                                    gradients[pass_key] = [gradient_dict[pass_key], gradients[pass_key]]
                            else:
                                gradients[pass_key] = gradient_dict[pass_key]

                        # print('\n Gradients after: ', gradients)

                        sent_data = {'action':ActionTypes.BACKWARD,
                                    'forward_pass_id':forward_pass_id, 
                                    'data':gradients, 
                                    }
                        t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.BACKWARD, self.backward_target_host, self.backward_target_port,))
                        send_trigger_threads.append(t)
                        t.start()
                        # self.trigger_send(type=ActionTypes.BACKWARD, target_host=self.backward_target_host, target_port=self.backward_target_port)
                    
                    if self.input_tensors.get(forward_pass_id, None) is not None:
                        del self.input_tensors[forward_pass_id]

                    print('Backward done, Used RAM %: ', psutil.virtual_memory().percent)
                    self.n_backwards += 1

                    if self.n_backwards % self.reduce_threshold == 0:
                        self.parallel_ring_reduce()

            self.node_status = NodeStatus.IDLE

            if len(self.load_forward_buffer) != 0:
                self.forward_lock.acquire(block=True)
                value = self.load_forward_buffer[0]
                del self.load_forward_buffer[0]
                self.forward_lock.release()
                action = value['action']

                if action == ActionTypes.FORWARD and self.node_type == NodeTypes.LEAF:
                    action = ActionTypes.FIND_LOSS
                if action == ActionTypes.NO_GRAD_FORWARD and self.node_type == NodeTypes.LEAF:
                    action = ActionTypes.ACCURACY
                
                if action == ActionTypes.FORWARD:
                    self.node_status = NodeStatus.FORWARD
                    data = value['data']
                    forward_pass_id = value['forward_pass_id']
                    model_args = self.create_model_args(data, forward_pass_id=forward_pass_id)
                    # print('model args: ', model_args)
                    
                    if not self.model.training:
                        self.model.train()

                    output = self.model(*model_args)

                    payload = self.create_forward_payload(output)


                    # print('OUT: ', type(output), len(output))
                    
                    final_payload = data
                    final_payload[self.submod_file] = payload

                    # self.send_buffer.append({'data_id':value['data_id'],
                    #                         'forward_pass_id':forward_pass_id,
                    #                         'data': final_payload,
                    #                         'input_size': value['input_size'],
                    #                         'action': ActionTypes.FIND_LOSS})
                    sent_data = {'data_id':value['data_id'],
                                'forward_pass_id':forward_pass_id,
                                'data': final_payload,
                                'input_size': value['input_size'],
                                'action': ActionTypes.FIND_LOSS}
                    t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
                    send_trigger_threads.append(t)
                    t.start()

                    # self.trigger_send(type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
                    print('Forward Done Used RAM %: ', psutil.virtual_memory().percent)

                elif action == ActionTypes.FIND_LOSS:
                    print('In find loss')
                    self.node_status = NodeStatus.FORWARD
                    data_id = value['data_id']
                    if isinstance(self.labels, torch.Tensor):
                        targets = self.labels[data_id:data_id+value['input_size']]
                    else:
                        targets = next(itertools.islice(self.labels, data_id, None))[1]

                    # if hasattr(self.labels, '__iter__'):
                    #     targets = next(itertools.islice(self.labels, data_id, None))[1]
                    # else:
                    #     targets = self.labels[data_id:data_id+value['input_size']]
                    
                    data = value['data']

                    # print('data in find_loss: ', data)

                    model_args = self.create_model_args(data)
                    # print('model args: ', model_args)

                    if not self.model.training:
                        self.model.train()

                    outputs = self.model(*model_args.values())
                    
                    loss = self.criterion(outputs, targets)

                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    gradients = self.create_backward_payload(model_args=model_args)

                    # print('shape of gradients: ', gradients.shape, value['tensor_id'])

                    # self.send_buffer.append({'action':ActionTypes.BACKWARD, 
                    #                          'data':gradients, 
                    #                          'forward_pass_id':value['forward_pass_id'],
                    #                          })
                    sent_data = {'action':ActionTypes.BACKWARD, 
                                'data':gradients, 
                                'forward_pass_id':value['forward_pass_id'],
                                }
                    t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.BACKWARD, self.backward_target_host, self.backward_target_port,))
                    send_trigger_threads.append(t)
                    t.start()

                    # self.trigger_send(type=ActionTypes.BACKWARD, target_host=self.backward_target_host, target_port=self.backward_target_port)
                    print('Find loss done Used RAM %: ', psutil.virtual_memory().percent)
                    self.n_backwards += 1

                    if self.n_backwards % self.reduce_threshold == 0:
                        self.parallel_ring_reduce()

                elif action == ActionTypes.NO_GRAD_FORWARD:
                    self.parallel_ring_reduce()
                    self.node_status = NodeStatus.FORWARD
                    print('No grad forward')
                    data = value['data']
                    model_args = self.create_no_grad_model_args(data)
                    self.model.eval()            
                    with torch.no_grad():
                        output = self.model(*model_args)

                    payload = self.create_no_grad_forward_payload(output)

                    final_payload = data
                    final_payload[self.submod_file] = payload

                    # self.send_buffer.append({
                    #                         'data': final_payload,
                    #                         'action': value['output_type']                                            
                    #                         })
                    sent_data = {
                                    'data': final_payload,
                                    'action': value['output_type']                                            
                                }
                    t = Thread(target=self.trigger_send, args=(sent_data, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
                    send_trigger_threads.append(t)
                    t.start()

                    # self.trigger_send(type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)

                elif action == ActionTypes.ACCURACY:
                    self.parallel_ring_reduce()
                    print('Finding accuracy')
                    data = value['data']
                    model_args = self.create_no_grad_model_args(data)
        
                    self.model.eval()
                    with torch.no_grad():
                        y_pred = self.model(*model_args)
                        y_pred = np.argmax(y_pred.detach().numpy(), axis=-1)
                        y_test = np.argmax(self.test_labels, axis=-1)
                        accuracy = np.sum(y_pred == y_test, axis=0)/len(y_test)
                        print('\nTest Accuracy: ', accuracy)

                elif action == ActionTypes.PREDICTION:
                    data = value['data']
                    print('Prediction: ', data)
                    model_args = self.create_no_grad_model_args(data)
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(*model_args)
                    print('Predicted: ', pred)

                elif action == ActionTypes.SAVE_SUBMODEL:
                    script = torch.jit.script(self.model)
                    script.save('trained_submodels/{}.pt'.format(self.submod_file))
                    if self.node_type != NodeTypes.LEAF:
                        # self.send_buffer.append({'action': ActionTypes.SAVE_SUBMODEL})
                        t = Thread(target=self.trigger_send, args=({'action': ActionTypes.SAVE_SUBMODEL}, ActionTypes.FORWARD, self.forward_target_host, self.forward_target_port,))
                        send_trigger_threads.append(t)
                        t.start()
                        # self.trigger_send(type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
                    print('SAVE done')

            if len(send_trigger_threads)>0:
                for send_threads in send_trigger_threads:
                    send_threads.join()
            self.node_status = NodeStatus.IDLE
                        

    def forward_compute(self, data_id=None, tensors=None):
        self.node_status = NodeStatus.FORWARD
        if not self.model.training:
            self.model.train()

        output = self.model(tensors)

        payload = self.create_forward_payload(output, tensors=tensors)

        # print('OUT: ', type(output), len(output))
        
        final_payload = {}
        final_payload[self.submod_file] = payload

        # self.send_buffer.append({'data_id':data_id,
        #                          'forward_pass_id':self.forward_pass_id,
        #                         'data': final_payload,
        #                         'input_size': tensors.shape[0],
        #                         'action': ActionTypes.FORWARD})
        sent_data = {'data_id':data_id,
                    'forward_pass_id':self.forward_pass_id,
                    'data': final_payload,
                    'input_size': tensors.shape[0],
                    'action': ActionTypes.FORWARD}
        
        self.forward_pass_id += 1
        self.trigger_send(sent_data, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('Forward compute done for: ', self.tensor_id)
        self.node_status = NodeStatus.IDLE

        

    def no_grad_forward_compute(self, tensors=None, output_type=None):
        self.parallel_ring_reduce()
        self.node_status = NodeStatus.FORWARD
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensors)

        payload = self.create_no_grad_forward_payload(output, tensors=tensors)

        final_payload = {}
        final_payload[self.submod_file] = payload

        # self.send_buffer.append({
        #                         'data': final_payload,
        #                         'action': ActionTypes.NO_GRAD_FORWARD,
        #                         'output_type': output_type
        #                         })
        sent_data = {
                        'data': final_payload,
                        'action': ActionTypes.NO_GRAD_FORWARD,
                        'output_type': output_type
                    }
        self.trigger_send(sent_data, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('No Grad forward compute done')
        self.node_status = NodeStatus.IDLE

    
    def create_model_args(self, data, forward_pass_id=None):
        # print('\ndata received: ', data)
        # print('\ninput templates: ', self.input_template)
        if self.node_type != NodeTypes.LEAF:
            model_args = []
            self.input_tensors[forward_pass_id] = {}
            for arg_pos, arg_metadata in self.input_template.items():
                for k, v in arg_metadata.items():

                    if isinstance(v, str) or isinstance(v, int):
                        print(k, data.keys())
                        if isinstance(v, int):
                            arg_pos = v
                        else:
                            arg_pos = 0
                        
                        if self.submod_file in data[k][arg_pos]['target']:
                            tensor_id = data[k][arg_pos]['tensor_id']

                            if data[k][arg_pos]['data'].device.type != self.device:
                                data[k][arg_pos]['data'] = data[k][arg_pos]['data'].to(self.device)

                            model_args.append(data[k][arg_pos]['data'])
                            if self.node_type != NodeTypes.LEAF:
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
                                if self.node_type != NodeTypes.LEAF:
                                    if k != 'model_inputs':
                                        self.input_tensors[forward_pass_id][tensor_id] = data[k][v]['data']                                 
                            # elif 'placeholder' in v:
                            #     model_args.append(data[k][0]['data'])
                            #     if self.node_type != 'leaf':
                            #         self.input_tensors[forward_pass_id][tensor_id] = data[k][0]['data']

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
                            if self.node_type != NodeTypes.LEAF:
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
                            # elif 'placeholder' in v:
                            #     model_args[tensor_id] = data[k][0]['data']

                        data[k][v]['target'].remove(self.submod_file)

                        if len(data[k][v]['target']) == 0:
                            del data[k][v]
                        
                        if len(data[k]) == 0:
                            del data[k]
        # print('\n model args: ', model_args)
        # print('\ninput tensors: ', self.input_tensors)
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
                        # elif 'placeholder' in v:
                        #     model_args.append(data[k][0]['data'])
            
                    data[k][v]['target'].remove(self.submod_file)

                    if len(data[k][v]['target']) == 0:
                        del data[k][v]
                    
                    if len(data[k]) == 0:
                        del data[k]
            
        return model_args

    def create_backward_payload(self, forward_pass_id=None, model_args=None):        
        grad_payload = {}
        if self.node_type == NodeTypes.LEAF:
            for key, value in model_args.items():
                if value.requires_grad:
                    grad_payload[key] = value.grad
        else:
            for key, value in self.input_tensors[forward_pass_id].items():
                if value.requires_grad:
                    grad_payload[key] = value.grad
        return grad_payload

    def create_forward_payload(self, output, tensors=None):
        payload = self.output_template.copy()
        # print('\npayload in forward payload ', payload, output)
        for k, v in payload.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output
            payload[k]['data'] = out
            payload[k]['tensor_id'] = self.tensor_id
            self.output_tensors[self.tensor_id] = out
            # self.tensor_id += 1
            self.tensor_id = str(int(self.tensor_id.split('_')[0]) + 1) + '_{}'.format(self.submod_file)

        if self.node_type == NodeTypes.ROOT:
            payload['model_inputs'] = self.model_inputs_template
            for k, v in self.model_inputs_template.items():
                if payload['model_inputs'][k].get('target', None) is not None:
                    payload['model_inputs'][k]['data'] = tensors[k]
        # print('output tensors: ', self.output_tensors)
        return payload

    def create_no_grad_forward_payload(self, output, tensors=None):
        payload = self.output_template.copy()
        for k, v in payload.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output
            payload[k]['data'] = out
            
        if self.node_type == NodeTypes.ROOT:
            payload['model_inputs'] = self.model_inputs_template
            for k, v in self.model_inputs_template.items():
                if payload['model_inputs'][k].get('target', None) is not None:
                    payload['model_inputs'][k]['data'] = tensors[k]

        return payload

    def trigger_save_submodel(self):
        script = torch.jit.script(self.model)
        script.save('trained_submodels/{}.pt'.format(self.submod_file))
        # self.send_buffer.append({'action': ActionTypes.SAVE_SUBMODEL})
        self.trigger_send({'action': ActionTypes.SAVE_SUBMODEL}, type=ActionTypes.FORWARD, target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('SAVE done')

    # def trigger_send(self, type=None, target_host=None, target_port=None):

    #     if len(self.send_buffer) > 0:
    #         with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
    #             stub = CommServerStub(channel)

    #             send_flag = False
    #             while not send_flag:
    #                 buffer_status = stub.buffer_status(CheckBufferStatus(name=self.name, type=type))
                    
    #                 if buffer_status.status == BufferStatus.SEND_BUFFER:
    #                     send_flag = True
                 

    #             response = stub.send_buffer(generate_stream(self.send_buffer[0], type=type))

    #             self.send_buffer = []
        
    def trigger_send(self, data, type=None, target_host=None, target_port=None):
        with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
            stub = CommServerStub(channel)

            send_flag = False
            while not send_flag:
                buffer_status = stub.buffer_status(CheckBufferStatus(name=self.name, type=type))
                
                if buffer_status.status == BufferStatus.SEND_BUFFER:
                    send_flag = True
                

            response = stub.send_buffer(generate_stream(data, type=type))


    def parallel_ring_reduce(self):#, data_dict):
        # print('\n Rank: ', rank, 'Ring ids: ', ring_ids, ' data dict: ', data_dict)
        print('\nBegining Parameter Averaging')
        # print('State dict before averaging: ', self.model.state_dict())
        threads = []
        # data_dict_keys = list(data_dict.keys())
        for ring_id, _ in self.ring_ids.items():
            ring_data = {k:self.model.state_dict()[k] for k in self.ring_param_keys[ring_id]}
            t = Thread(target=self.single_ring_reduce, args=(ring_data,ring_id,))
            threads.append(t)
            t.start()

        # for thread in threads:
        #     thread.start()

        for thread in threads:
            thread.join()

        # print('Averaged params buffer: ', len(self.averaged_params_buffer), self.averaged_params_buffer.keys())

        # self.model.load_state_dict(self.averaged_params_buffer, strict=False)
        load_state_dict_conserve_versions(self.model, self.averaged_params_buffer)

        load_model_weights_into_optim(self.model, self.optimizer)
        # print('Optimizer weights updated: ', self.optimizer.state)
        self.average_no += 1 
        print('\nParameter Averaging Complete: ', self.average_no, ' Used RAM %: ', psutil.virtual_memory().percent)

    def single_ring_reduce(self, ring_data, ring_id):
        # print('Starting ring reduce thread for: ', ring_id, ring_data)

        chunked_data = create_chunks(data=ring_data, size=self.ring_size)

        # for param, c in chunked_data.items():
        #     print('p: ', param)
        #     for ch in c['data']:
        #         print(' chunk shape', ch.shape)

        iterations = self.ring_size - 1

        recv_pos = ((self.rank-1)+self.ring_size) % self.ring_size 
        send_pos = (self.rank)%self.ring_size
        # print('send pos: ', send_pos)
        for i in range(iterations):
            address_send_data_mapping = {}
            for id, chunks in chunked_data.items():
                dest = self.param_address_mapping[id]
                # print(' Dest: ', dest, ' send pos: ', send_pos, ' chunk: ', chunks[send_pos].shape)
                if dest in address_send_data_mapping:
                    address_send_data_mapping[dest][id] = {'pos':send_pos, 'chunk':chunks['data'][send_pos]}
                else:
                    address_send_data_mapping[dest] = {id:{'pos':send_pos, 'chunk': chunks['data'][send_pos]}}
            
            # print(address_send_data_mapping)
            send_threads = []
            for address, data_dict in address_send_data_mapping.items():
                t = Thread(target=self.send_reduce_chunk, args=(address.split(':')[0], address.split(':')[1], data_dict, ring_id,))
                send_threads.append(t)
                t.start()

            for t in send_threads:
                t.join()
            send_threads = []
            keys_received = 0
            # print('chunked data length: ', len(chunked_data))
            while keys_received < len(chunked_data):
                with self.reduce_lock:
                    received_data = self.reduce_ring_buffers.get(ring_id, None)
                    if received_data is not None and len(received_data)>0:
                        # print('Received from reduce buffer: ', received_data)
                        recv_chunk = received_data.pop(0)
                        # print('Recv chunk in node: ', recv_chunk)
                        self.reduce_ring_buffers[ring_id] = received_data
                        for param_index, chunk_dict in recv_chunk.items():
                            # print('param: ', param_index, ' recv pos: ', recv_pos, ' pos from data: ' , chunk_dict['pos'], ' received chunk: ', chunk_dict['chunk'].shape)
                            # chunked_data[param_index]['data'][chunk_dict['pos']] += chunk_dict['chunk'][:]
                            chunked_data[param_index]['data'][chunk_dict['pos']] = chunked_data[param_index]['data'][chunk_dict['pos']].add(chunk_dict['chunk'][:])
                            keys_received += 1
            
            recv_pos = ((recv_pos - 1)+self.ring_size)%self.ring_size
            send_pos = ((send_pos - 1)+self.ring_size)%self.ring_size
            self.reduce_iteration[ring_id] += 1

        self.reduce_iteration[ring_id] = 0
        # print('Ring id: ', ring_id, ' Reduced: ', chunked_data, ' reduce iteration reset: ', self.reduce_iteration[ring_id])
        print('Reduced Ring id: ', ring_id)

        send_pos = (recv_pos+1)%self.ring_size
        recv_pos = ((send_pos - 1)+self.ring_size)%self.ring_size
        # time.sleep(5)
        for i in range(iterations):
            address_send_data_mapping = {}
            for id, chunks in chunked_data.items():
                dest = self.param_address_mapping[id]
                # if dest in address_send_data_mapping:
                #     address_send_data_mapping[dest][id] = chunks[send_pos]
                # else:
                #     address_send_data_mapping[dest] = {id:chunks[send_pos]}

                if dest in address_send_data_mapping:
                    address_send_data_mapping[dest][id] = {'pos':send_pos, 'chunk':chunks['data'][send_pos]}
                else:
                    address_send_data_mapping[dest] = {id:{'pos':send_pos, 'chunk': chunks['data'][send_pos]}}

            send_threads = []
            for address, data_dict in address_send_data_mapping.items():
                t = Thread(target=self.send_gather_chunk, args=(address.split(':')[0], address.split(':')[1], data_dict, ring_id,))
                send_threads.append(t)
                t.start()

            for t in send_threads:
                t.join()
                
            keys_received = 0
            while keys_received < len(chunked_data):
                with self.gather_lock:
                    received_data = self.gather_ring_buffers.get(ring_id, None)
                    if received_data is not None and len(received_data)>0:
                        # print('Received data in node: ', received_data)
                        recv_chunk = received_data.pop(0)
                        # print('Recv chunk in node: ', recv_chunk)
                        self.gather_ring_buffers[ring_id] = received_data
                        
                        for param_index, chunk_dict in recv_chunk.items():
                            # chunked_data[param_index][recv_pos] = chunk[:]
                            chunked_data[param_index]['data'][chunk_dict['pos']].data = chunk_dict['chunk'].data[:]
                            # chunked_data[param_index]['data'][chunk_dict['pos']] = chunk_dict['chunk'][:]
                            keys_received += 1

            recv_pos = ((recv_pos - 1)+self.ring_size)%self.ring_size
            send_pos = ((send_pos - 1)+self.ring_size)%self.ring_size
            self.gather_iteration[ring_id] += 1

        self.gather_iteration[ring_id] = 0
        # print('Ring id: ', ring_id,'Gathered: {}'.format(chunked_data))

        for param, chunk in chunked_data.items():
            chunked_data[param] = torch.cat(chunk['data'], dim=chunk['split_axis']).div(self.ring_size)

        # print('Ring id: ', ring_id,'Gathered after cat: {}'.format(chunked_data))
        print('Gathered Ring id: ', ring_id)
        self.averaged_params_buffer.update(chunked_data)

    def send_reduce_chunk(self, target_host, target_port, data_dict, ring_id):
        # print('target host: ', target_host, target_port)
        with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
            stub = CommServerStub(channel)
            while True:
                iteration_resp = stub.reduce_iteration(CheckReduceIteration(ring_id=ring_id))
                if iteration_resp.iteration == self.reduce_iteration[ring_id]:
                    break
            response = stub.reduce_chunk(generate_data_stream(data_dict, ring_id=ring_id))

    def send_gather_chunk(self, target_host, target_port, data_dict, ring_id):
        # print('target host: ', target_host, target_port)
        with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
            stub = CommServerStub(channel)
            while True:
                iteration_resp = stub.gather_iteration(CheckGatherIteration(ring_id=ring_id))
                if iteration_resp.iteration == self.gather_iteration[ring_id]:
                    break
            response = stub.gather_chunk(generate_data_stream(data_dict, ring_id=ring_id))


    def __getstate__(self):
        return dict(
            forward_lock = self.forward_lock,
            backward_lock = self.backward_lock,
            reduce_lock = self.reduce_lock,
            gather_lock = self.gather_lock,
            local_address = self.local_address,
            load_forward_buffer = self.load_forward_buffer,
            load_backward_buffer = self.load_backward_buffer,
            reduce_ring_buffers = self.reduce_ring_buffers,
            gather_ring_buffers = self.gather_ring_buffers,
            reduce_iteration = self.reduce_iteration,
            gather_iteration = self.gather_iteration
        )

    def __setstate__(self, state):
        self.local_address = state['local_address']
        self.init_server(load_forward_buffer=state['load_forward_buffer'], 
                         load_backward_buffer=state['load_backward_buffer'], 
                         reduce_ring_buffers= state['reduce_ring_buffers'],
                         gather_ring_buffers= state['gather_ring_buffers'],
                         forward_lock=state['forward_lock'], 
                         backward_lock=state['backward_lock'],
                         reduce_lock=state['reduce_lock'],
                         gather_lock=state['gather_lock'],
                         reduce_iteration = state['reduce_iteration'],
                         gather_iteration = state['gather_iteration']
                         )

def create_chunks(data, size):
    chunked_data = {}
    for key, val in data.items():
        print(key, val.shape)
        split_axis = np.argmax(val.shape)
        chunked_data[key] = {}
        # chunked_data[key]['data'] = np.array_split(val, size, axis=split_axis)
        chunked_data[key]['data'] = list(torch.chunk(val, chunks=size, dim=split_axis))
        chunked_data[key]['split_axis'] = split_axis
        # chunked_data[key] = np.array_split(val, size, axis=split_axis)

    return chunked_data