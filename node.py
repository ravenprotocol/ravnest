from concurrent import futures
import asyncio
import grpc
import threading
import multiprocessing as mp
from threading import Thread
import numpy as np
import pickle
import time
from utils import *
from endpoints import GrpcService
from server_pb2_grpc import add_CommServerServicer_to_server, CommServerStub
from server_pb2 import CheckBufferStatus

class Node():
    def __init__(self, name=None, 
                 submod_file=None, template_path=None, 
                 local_host=None, local_port=None,
                 forward_target_host=None, forward_target_port=None, 
                 backward_target_host=None, backward_target_port=None, 
                 model=None, optimizer=None, labels=None, test_labels=None,
                 ring_ids=None, rank=None, ring_size = None, data_dict = None,
                 param_addresses = None
                 ):
        self.manager = mp.Manager()
        self.forward_lock = mp.Lock()
        self.backward_lock = mp.Lock()
        self.reduce_lock = mp.Lock()
        self.gather_lock = mp.Lock()
        
        self.local_address = '{}:{}'.format(local_host, local_port)
        self.name = name

        self.load_forward_buffer = self.manager.list()
        self.load_backward_buffer = self.manager.list()
        self.reduce_ring_buffers = self.manager.dict()
        self.gather_ring_buffers = self.manager.dict()
        self.ring_ids = ring_ids
        self.rank = rank
        self.ring_size = ring_size
        
        self.ring_param_keys = {}
        self.data_dict = data_dict
        if data_dict is not None:
            data_dict_keys = list(data_dict.keys())
            for i, ring in enumerate(self.ring_ids.items()):
                if i < len(self.ring_ids) - 1:
                    keys = data_dict_keys[data_dict_keys.index(ring[1]):data_dict_keys.index(self.ring_ids[ring[0]+1])]
                else:
                    keys = data_dict_keys[data_dict_keys.index(ring[1]):]
                
                self.ring_param_keys[ring[0]] = keys

            self.param_address_mapping = {}
            for i, address_to_param in enumerate(param_addresses.items()):
                if i < len(param_addresses) - 1:
                    keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):data_dict_keys.index(param_addresses[list(param_addresses.keys())[i+1]])]
                else:
                    keys = data_dict_keys[data_dict_keys.index(address_to_param[1]):]
                
                for param_name in keys:
                    self.param_address_mapping[param_name] = address_to_param[0]

        self.send_buffer = []
        self.model = model
        self.labels = labels
        self.test_labels = test_labels
        self.forward_target_host = forward_target_host
        self.forward_target_port = forward_target_port
        self.backward_target_host = backward_target_host
        self.backward_target_port = backward_target_port
        self.tensor_id = 0
        self.output_tensors = {}
        self.input_tensors = {}
        self.n_backwards = 0
        self.forward_pass_id = 0

        self.submod_file = submod_file

        if submod_file is not None:
            with open('{}/{}_input.pkl'.format(template_path, submod_file), 'rb') as fout:
                self.input_template = pickle.load(fout)

            with open('{}/{}_output.pkl'.format(template_path, submod_file), 'rb') as fout:
                self.output_template = pickle.load(fout)

            if self.backward_target_host is None and self.backward_target_port is None:
                self.node_type = 'root'
                with open('{}/model_inputs.pkl'.format(template_path), 'rb') as fout:
                    self.model_inputs_template = pickle.load(fout)
                self.optimizer = optimizer(current_model_params_clone(self.model))
            elif self.forward_target_host is None and self.forward_target_port is None:
                self.node_type = 'leaf'
                self.optimizer = optimizer(self.model.parameters())
            else:
                self.node_type = 'mid'
                self.optimizer = optimizer(current_model_params_clone(self.model))

    def init_server(self, load_forward_buffer=None, load_backward_buffer=None, 
                    reduce_ring_buffers = None, gather_ring_buffers = None, 
                    forward_lock=None, backward_lock=None, reduce_lock=None, gather_lock=None):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_CommServerServicer_to_server(GrpcService(
            load_forward_buffer=load_forward_buffer, load_backward_buffer=load_backward_buffer, 
            reduce_ring_buffers=reduce_ring_buffers, gather_ring_buffers=gather_ring_buffers,
            forward_lock=forward_lock, backward_lock=backward_lock, reduce_lock=reduce_lock, gather_lock=gather_lock), self.server)
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
            if len(self.load_backward_buffer) != 0:
                self.backward_lock.acquire(block=True)
                value = self.load_backward_buffer[0]
                del self.load_backward_buffer[0]
                self.backward_lock.release()
                action = value['action']

                if action == 'backward':
                    gradient_dict = value['data']
                    forward_pass_id = value['forward_pass_id']
                    
                    self.model.zero_grad()
                    self.optimizer.zero_grad()

                    for key, value in gradient_dict.items():
                        output_tensor = self.output_tensors[key]
                                    
                        if len(self.output_tensors) > 1:
                            output_tensor.backward(value, retain_graph=True)
                        else:
                            output_tensor.backward(value)

                        del self.output_tensors[key]

                    load_grads_into_optimizer(self.model, self.optimizer)
                    self.optimizer.step()
                    load_optim_weights_into_model(self.model, self.optimizer)

                    if self.node_type != 'root':
                        gradients = self.create_backward_payload(forward_pass_id=forward_pass_id)
                        self.send_buffer.append({'action':'backward', 
                                                'forward_pass_id':forward_pass_id, 
                                                'data':gradients, 
                                                })

                        self.trigger_send(type='backward', target_host=self.backward_target_host, target_port=self.backward_target_port)
                    
                    if self.input_tensors.get(forward_pass_id, None) is not None:
                        del self.input_tensors[forward_pass_id]

                    print('Backward done')
                    self.n_backwards += 1

            if len(self.load_forward_buffer) != 0:
                self.forward_lock.acquire(block=True)
                value = self.load_forward_buffer[0]
                del self.load_forward_buffer[0]
                self.forward_lock.release()
                action = value['action']
                
                if action == 'forward':
                    data = value['data']
                    forward_pass_id = value['forward_pass_id']
                    model_args = self.create_model_args(data, forward_pass_id=forward_pass_id)
                                
                    if not self.model.training:
                        self.model.train()

                    output = self.model(*model_args)

                    payload = self.create_forward_payload(output)


                    # print('OUT: ', type(output), len(output))
                    
                    final_payload = data
                    final_payload[self.submod_file] = payload

                    self.send_buffer.append({'data_id':value['data_id'],
                                            'forward_pass_id':forward_pass_id,
                                            'data': final_payload,
                                            'input_size': value['input_size'],
                                            'action': 'find_loss'})
                    
                    self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)
                    print('Forward Done')

                elif action == 'find_loss':
                    data_id = value['data_id']
                    targets = self.labels[data_id:data_id+value['input_size']]
                    
                    data = value['data']

                    # print('data in find_loss: ', data)

                    model_args = self.create_model_args(data)


                    if not self.model.training:
                        self.model.train()

                    outputs = self.model(*model_args.values())

                    # loss = torch.nn.functional.mse_loss(outputs, targets)
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1)

                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    gradients = self.create_backward_payload(model_args=model_args)

                    # print('shape of gradients: ', gradients.shape, value['tensor_id'])

                    self.send_buffer.append({'action':'backward', 
                                             'data':gradients, 
                                             'forward_pass_id':value['forward_pass_id'],
                                             })
                    self.trigger_send(type='backward', target_host=self.backward_target_host, target_port=self.backward_target_port)
                    print('Find loss done')
                    self.n_backwards += 1

                elif action == 'no_grad_forward':
                    print('No grad forward')
                    data = value['data']
                    model_args = self.create_no_grad_model_args(data)
                    self.model.eval()            
                    with torch.no_grad():
                        output = self.model(*model_args)

                    payload = self.create_no_grad_forward_payload(output)

                    final_payload = data
                    final_payload[self.submod_file] = payload

                    self.send_buffer.append({
                                            'data': final_payload,
                                            'action': value['output_type']                                            
                                            })

                    self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)

                elif action == 'accuracy':
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

                elif action == 'prediction':
                    data = value['data']
                    print('Prediction: ', data)
                    model_args = self.create_no_grad_model_args(data)
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(*model_args)
                    print('Predicted: ', pred)

                elif action == 'save_submodel':
                    script = torch.jit.script(self.model)
                    script.save('trained_submodels/{}.pt'.format(self.submod_file))
                    if self.node_type != 'leaf':
                        self.send_buffer.append({'action': 'save_submodel'})
                        self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)
                    print('SAVE done')
                        

    def forward_compute(self, data_id=None, tensors=None):
        
        if not self.model.training:
            self.model.train()

        output = self.model(tensors)

        payload = self.create_forward_payload(output, tensors=tensors)

        # print('OUT: ', type(output), len(output))
        
        final_payload = {}
        final_payload[self.submod_file] = payload

        self.send_buffer.append({'data_id':data_id,
                                 'forward_pass_id':self.forward_pass_id,
                                'data': final_payload,
                                'input_size': tensors.shape[0],
                                'action': 'forward'})

        self.forward_pass_id += 1
        self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('Forward compute done for: ', self.tensor_id)
        

    def no_grad_forward_compute(self, tensors=None, output_type=None):

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensors)

        payload = self.create_no_grad_forward_payload(output, tensors=tensors)

        final_payload = {}
        final_payload[self.submod_file] = payload

        self.send_buffer.append({
                                'data': final_payload,
                                'action': 'no_grad_forward',
                                'output_type': output_type
                                })

        self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('No Grad forward compute done')

              
    def create_model_args(self, data, forward_pass_id=None):
        if self.node_type != 'leaf':
            model_args = []
            self.input_tensors[forward_pass_id] = {}
            for arg_pos, arg_metadata in self.input_template.items():
                for k, v in arg_metadata.items(): 

                    if isinstance(v, str) or isinstance(v, int):
                        if self.submod_file in data[k][arg_pos]['target']:
                            tensor_id = data[k][arg_pos]['tensor_id']
                            model_args.append(data[k][arg_pos]['data'])
                            if self.node_type != 'leaf':
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
                                model_args.append(data[k][v]['data'])   
                                if self.node_type != 'leaf':
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
                        if self.submod_file in data[k][arg_pos]['target']:
                            tensor_id = data[k][arg_pos]['tensor_id']
                            model_args[tensor_id] = data[k][arg_pos]['data']
                            if self.node_type != 'leaf':
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
                                model_args[tensor_id] = data[k][v]['data']    
                            # elif 'placeholder' in v:
                            #     model_args[tensor_id] = data[k][0]['data']

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
                    if self.submod_file in data[k][arg_pos]['target']:
                        model_args.append(data[k][arg_pos]['data'])

                        data[k][arg_pos]['target'].remove(self.submod_file)

                        if len(data[k][arg_pos]['target']) == 0:
                            del data[k][arg_pos]
                        
                        if len(data[k]) == 0:
                            del data[k]
                    
                       
                elif self.submod_file in data[k][v]['target']:
                    if 'submod' in k or 'model_inputs' in k:                                    
                        if isinstance(v, int):                                        
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
        if self.node_type == 'leaf':
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
        for k, v in payload.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output
            payload[k]['data'] = out
            payload[k]['tensor_id'] = self.tensor_id
            self.output_tensors[self.tensor_id] = out
            self.tensor_id += 1

        if self.node_type == 'root':
            payload['model_inputs'] = self.model_inputs_template
            for k, v in self.model_inputs_template.items():
                if payload['model_inputs'][k].get('target', None) is not None:
                    payload['model_inputs'][k]['data'] = tensors[k]

        return payload

    def create_no_grad_forward_payload(self, output, tensors=None):
        payload = self.output_template.copy()
        for k, v in payload.items():
            if isinstance(output, tuple):
                out = output[k]
            else:
                out = output
            payload[k]['data'] = out
            
        if self.node_type == 'root':
            payload['model_inputs'] = self.model_inputs_template
            for k, v in self.model_inputs_template.items():
                if payload['model_inputs'][k].get('target', None) is not None:
                    payload['model_inputs'][k]['data'] = tensors[k]

        return payload

    def trigger_save_submodel(self):
        script = torch.jit.script(self.model)
        script.save('trained_submodels/{}.pt'.format(self.submod_file))
        self.send_buffer.append({'action': 'save_submodel'})
        self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('SAVE done')

    def trigger_send(self, type=None, target_host=None, target_port=None):

        if len(self.send_buffer) > 0:
            with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
                stub = CommServerStub(channel)

                send_flag = False
                while not send_flag:
                    buffer_status = stub.buffer_status(CheckBufferStatus(name=self.name, type=type))
                    
                    if buffer_status.status == 'send_buffer':
                        send_flag = True
                 

                response = stub.send_buffer(generate_stream(self.send_buffer[0], type=type))

                self.send_buffer = []

    def parallel_ring_reduce(self):#, data_dict):
        # print('\n Rank: ', rank, 'Ring ids: ', ring_ids, ' data dict: ', data_dict)
        threads = []
        # data_dict_keys = list(data_dict.keys())
        for ring_id, index in self.ring_ids.items():
            # if i < len(self.ring_ids) - 1:
            #     keys = data_dict_keys[data_dict_keys.index(ring[1]):data_dict_keys.index(self.ring_ids[ring[0]+1])]
            # else:
            #     keys = data_dict_keys[data_dict_keys.index(ring[1]):]
            # # if device_name == 'c0_a':
            # #     print('keys in rank: ', rank, ': ', keys)
            ring_data = {k:self.data_dict[k] for k in self.ring_param_keys[ring_id]}
            t = Thread(target=self.single_ring_reduce, args=(ring_data,ring_id,))#, args=(device_name, ring_data, rank, size, {k:send_qs[k] for k in keys}, {k:receive_qs[k] for k in keys}))
            threads.append(t)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def single_ring_reduce(self, ring_data, ring_id):
        print('Starting ring reduce thread')

        chunked_data = create_chunks(data=ring_data, size=self.ring_size)

        print('\nchunked data: ', chunked_data)  

        iterations = self.ring_size - 1

        recv_pos = ((self.rank-1)+self.ring_size) % self.ring_size 
        send_pos = (self.rank)%self.ring_size

        for i in range(iterations):
            address_send_data_mapping = {}
            for id, chunks in chunked_data.items():
                dest = self.param_address_mapping[id]
                if dest in address_send_data_mapping:
                    address_send_data_mapping[dest][id] = chunks[send_pos]
                else:
                    address_send_data_mapping[dest] = {id:chunks[send_pos]}
            
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
            while keys_received < len(chunked_data):
                received_data = self.reduce_ring_buffers.get(ring_id, None)
                if received_data is not None and len(received_data)>0:
                    print('Received from reduce buffer: ', received_data)
                    with self.reduce_lock:
                        recv_chunk = received_data.pop()
                    
                    for param_index, chunk in recv_chunk.items():
                        chunked_data[param_index][recv_pos] += chunk[:]
                        keys_received += 1


            #     if ring_id in self.reduce_ring_buffers:
            #         if len()
            # for id, chunks in chunked_data.items():
            #     rid, recv_data = recv_q[id].get()
            #     chunked_data[id][recv_pos] += recv_data[:]  
            
            recv_pos = ((recv_pos - 1)+self.ring_size)%self.ring_size
            send_pos = ((send_pos - 1)+self.ring_size)%self.ring_size

        print('Reduced: ', chunked_data)

        send_pos = (recv_pos+1)%self.ring_size
        recv_pos = ((send_pos - 1)+self.ring_size)%self.ring_size
        # time.sleep(5)
        for i in range(iterations):
            address_send_data_mapping = {}
            for id, chunks in chunked_data.items():
                dest = self.param_address_mapping[id]
                if dest in address_send_data_mapping:
                    address_send_data_mapping[dest][id] = chunks[send_pos]
                else:
                    address_send_data_mapping[dest] = {id:chunks[send_pos]}

            send_threads = []
            for address, data_dict in address_send_data_mapping.items():
                t = Thread(target=self.send_gather_chunk, args=(address.split(':')[0], address.split(':')[1], data_dict, ring_id,))
                send_threads.append(t)
                t.start()

            for t in send_threads:
                t.join()
            # for id, chunks in chunked_data.items():
            #     # sent_chunks[id] = chunks[send_pos]
            #     send_q[id].put((id, chunks[send_pos]))
                
            keys_received = 0
            while keys_received < len(chunked_data):
                received_data = self.gather_ring_buffers.get(ring_id, None)
                if received_data is not None and len(received_data)>0:
                    print('Received from Gather buffer: ', received_data)
                    with self.gather_lock:
                        recv_chunk = received_data.pop()
                    
                    for param_index, chunk in recv_chunk.items():
                        chunked_data[param_index][recv_pos] = chunk[:]
                        keys_received += 1

            # chunks[recv_pos] = recv_data[:]
            # for id, chunks in chunked_data.items():
            #     rid, recv_data = recv_q[id].get()
            #     # print('Received in Gather: ', device_name, ' for param id: ', rid, ' data: ', recv_data, ' should have received id: ', id, ' for pos: ', recv_pos)

            #     chunked_data[id][recv_pos] = recv_data[:]    

            recv_pos = ((recv_pos - 1)+self.ring_size)%self.ring_size
            send_pos = ((send_pos - 1)+self.ring_size)%self.ring_size


        print('Gathered: {}'.format(chunked_data)) 

    def send_reduce_chunk(self, target_host, target_port, data_dict, ring_id):
        print('target host: ', target_host, target_port)
        with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
            stub = CommServerStub(channel)
            response = stub.reduce_chunk(generate_data_stream(data_dict, ring_id=ring_id))

    def send_gather_chunk(self, target_host, target_port, data_dict, ring_id):
        print('target host: ', target_host, target_port)
        with grpc.insecure_channel('{}:{}'.format(target_host, target_port)) as channel:
            stub = CommServerStub(channel)
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
            gather_ring_buffers = self.gather_ring_buffers
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
                         )

def create_chunks(data, size):
    chunked_data = {}
    for key, val in data.items():
        split_axis = np.argmax(val.shape)
        chunked_data[key] = np.array_split(val, size, axis=split_axis)

    return chunked_data