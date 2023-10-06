from concurrent import futures
import asyncio
import grpc
import threading
import multiprocessing as mp
import numpy as np
import _pickle as cPickle
import time
from utils import *
from endpoints import GrpcService
from server_pb2_grpc import add_CommServerServicer_to_server, CommServerStub
from server_pb2 import CheckBufferStatus

class Node():
    def __init__(self, name=None, local_host=None, local_port=None,forward_target_host=None, forward_target_port=None, backward_target_host=None, backward_target_port=None, model=None, optimizer=None, labels=None, test_labels=None):
        self.manager = mp.Manager()
        self.forward_lock = mp.Lock()
        self.backward_lock = mp.Lock()
        
        self.local_address = '{}:{}'.format(local_host, local_port)
        self.name = name
        self.load_forward_buffer = self.manager.list()
        self.load_backward_buffer = self.manager.list()
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

        if self.backward_target_host is None and self.backward_target_port is None:
            self.node_type = 'root'
            self.optimizer = optimizer(current_model_params_clone(self.model))
        elif self.forward_target_host is None and self.forward_target_port is None:
            self.node_type = 'leaf'
            self.optimizer = optimizer(self.model.parameters())
        else:
            self.node_type = 'mid'
            self.optimizer = optimizer(current_model_params_clone(self.model))

    def init_server(self, load_forward_buffer=None, load_backward_buffer=None, forward_lock=None, backward_lock=None):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_CommServerServicer_to_server(GrpcService(load_forward_buffer=load_forward_buffer, load_backward_buffer=load_backward_buffer, forward_lock=forward_lock, backward_lock=backward_lock), self.server)


    def grpc_server_serve(self):
        print('Listening on : ', self.local_address)
        self.server.add_insecure_port(self.local_address)
        self.server.start()
        self.server.wait_for_termination()


    def start_grpc_server(self):
        asyncio.get_event_loop().run_until_complete(self.grpc_server_serve())

    def start(self):
        serve_process = mp.Process(target=self.grpc_server_serve, daemon=True)
        serve_process.start()
        time.sleep(2)
        buffer_thread = threading.Thread(target=self.check_load_forward_buffer, daemon=True)
        buffer_thread.start()

    def check_load_forward_buffer(self):
        while True:

            if len(self.load_backward_buffer) != 0:
                self.backward_lock.acquire(block=True)
                value = self.load_backward_buffer[0]
                del self.load_backward_buffer[0]
                self.backward_lock.release()
                action = value['action']

                if action == 'backward':
                    gradient = value['data']
                    tensor_id = value['tensor_id']
                    output_tensor = self.output_tensors[tensor_id]
                
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    if len(self.output_tensors) > 1:
                        output_tensor.backward(gradient, retain_graph=True)
                    else:
                        output_tensor.backward(gradient)

                    load_grads_into_optimizer(self.model, self.optimizer)
                    self.optimizer.step()
                    load_optim_weights_into_model(self.model, self.optimizer)

                    if self.node_type != 'root':
                        gradients = self.input_tensors[tensor_id].grad
                        self.send_buffer.append({'action':'backward', 
                                             'data':gradients, 
                                             'tensor_id':value['tensor_id']})

                        self.trigger_send(type='backward', target_host=self.backward_target_host, target_port=self.backward_target_port)
                    
                    del self.output_tensors[tensor_id]
                    if self.input_tensors.get(tensor_id, None) is not None:
                        del self.input_tensors[tensor_id]

                    print('Backward done for: ', tensor_id)
                    self.n_backwards += 1

            if len(self.load_forward_buffer) != 0:
                self.forward_lock.acquire(block=True)
                value = self.load_forward_buffer[0]
                del self.load_forward_buffer[0]
                self.forward_lock.release()
                action = value['action']
                
                if action == 'forward':
                    inputs = value['data']
                    tensor_id = value['tensor_id']
                    self.input_tensors[tensor_id] = inputs
                    output = self.model(inputs)
                    self.output_tensors[tensor_id] = output
                    self.send_buffer.append({'data_id':value['data_id'],
                                'tensor_id': tensor_id,
                                'data': output,
                                'input_size': value['input_size'],
                                'action': 'find_loss'})
                    self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)
                    print('Forward Done for: ', tensor_id)

                elif action == 'find_loss':
                    data_id = value['data_id']
                    targets = self.labels[data_id:data_id+value['input_size']]
                    inputs = value['data']
                    outputs = self.model(inputs)

                    loss = torch.nn.functional.mse_loss(outputs, targets)
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    gradients = inputs.grad

                    # print('shape of gradients: ', gradients.shape, value['tensor_id'])

                    self.send_buffer.append({'action':'backward', 
                                             'data':gradients, 
                                             'tensor_id':value['tensor_id']})
                    self.trigger_send(type='backward', target_host=self.backward_target_host, target_port=self.backward_target_port)
                    print('Find loss done for: ', value['tensor_id'])
                    self.n_backwards += 1

                elif action == 'no_grad_forward':
                    print('No grad forward')
                    ip = value['data']
                    with torch.no_grad():
                        out = self.model(ip)

                    self.send_buffer.append({'data':out, 'action':'find_accuracy'})
                    self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)

                elif action == 'find_accuracy':
                    print('Finding accuracy')
                    ip = value['data']
                    with torch.no_grad():
                        y_pred = self.model(ip)
                        y_pred = np.argmax(y_pred.detach().numpy(), axis=-1)
                        y_test = np.argmax(self.test_labels, axis=-1)
                        accuracy = np.sum(y_pred == y_test, axis=0)/len(y_test)
                        print('\nTest Accuracy: ', accuracy)


    def forward_compute(self, data_id=None, tensors=None):
        
        out = self.model(tensors)
        self.output_tensors[self.tensor_id] = out
        
        self.send_buffer.append({'data_id':data_id,
                                'tensor_id': self.tensor_id,
                                'data': out,
                                'input_size': tensors.shape[0],
                                'action': 'forward'})

        self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)
        print('Forward compute done for: ', self.tensor_id)
        self.tensor_id += 1

    def no_grad_forward_compute(self, tensors=None):
        with torch.no_grad():
            out = self.model(tensors)
        self.send_buffer.append({'data': out,'action': 'no_grad_forward'})
        self.trigger_send(type='forward', target_host=self.forward_target_host, target_port=self.forward_target_port)

        
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


    def __getstate__(self):
        return dict(
            forward_lock = self.forward_lock,
            backward_lock = self.backward_lock,
            local_address=self.local_address,
            load_forward_buffer=self.load_forward_buffer,
            load_backward_buffer=self.load_backward_buffer
        )

    def __setstate__(self, state):
        self.local_address = state['local_address']
        self.init_server(load_forward_buffer=state['load_forward_buffer'], 
                         load_backward_buffer=state['load_backward_buffer'], 
                         forward_lock=state['forward_lock'], 
                         backward_lock=state['backward_lock'])

