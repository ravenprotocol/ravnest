import torch
import torch.distributed as dist
import multiprocessing
import threading
from .strings import *
from .utils import *

mp = multiprocessing.get_context('spawn')

class Communication_TCP():
    def __init__(self, start_server_flag=None, node_type=None, backend='gloo', rank=None, world_size=None, 
                 forward_inner_pipe=None, backward_inner_pipe=None,
                 load_forward_buffer=None, load_backward_buffer=None,
                 forward_lock=None, backward_lock=None, input_tensors=None,
                 forward_input_shape=None, backward_input_shape=None):
        self.start_server_flag = start_server_flag
        self.backend = backend
        self.rank = rank
        self.node_type = node_type
        self.world_size = world_size
        self.forward_inner_pipe = forward_inner_pipe
        self.backward_inner_pipe = backward_inner_pipe
        self.load_forward_buffer = load_forward_buffer
        self.load_backward_buffer = load_backward_buffer
        self.forward_lock = forward_lock
        self.backward_lock = backward_lock
        self.forward_input_shape = forward_input_shape
        self.backward_input_shape = backward_input_shape
        self.input_tensors = input_tensors
    
    def forward_inner_pipe_monitor(self):
        while True:
            forward_tensor = self.forward_inner_pipe.recv()
            dist.send(forward_tensor, self.rank+1)

    def backward_inner_pipe_monitor(self):
        while True:
            forward_pass_id, grad_tensor = self.backward_inner_pipe.recv()
            dist.send(torch.tensor(forward_pass_id, dtype=torch.int16), self.rank-1)
            dist.send(grad_tensor, self.rank-1)

    def recv_forward_tensors(self):
        while True:
            if len(self.load_forward_buffer) == 0:
                forward_inputs = torch.zeros(self.forward_input_shape)
                dist.recv(forward_inputs, self.rank - 1)
                self.forward_lock.acquire(block=True)
                self.load_forward_buffer.append(forward_inputs)
                self.forward_lock.release()

    def recv_grad_tensors(self):
        while True:
            if len(self.load_backward_buffer) == 0:
                forward_pass_id = torch.zeros((), dtype=torch.int16)
                dist.recv(forward_pass_id, self.rank + 1)
                grad_inputs = torch.zeros(self.backward_input_shape)
                dist.recv(grad_inputs, self.rank + 1)
                self.backward_lock.acquire(block=True)
                self.load_backward_buffer.append((forward_pass_id, grad_inputs))
                self.backward_lock.release()

    def start(self):
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)

        threads = []
        if self.node_type == NodeTypes.ROOT or self.node_type == NodeTypes.STEM:
            forward_inner_pipe_monitor_thread = threading.Thread(target=self.forward_inner_pipe_monitor)
            threads.append(forward_inner_pipe_monitor_thread)
            forward_inner_pipe_monitor_thread.start()

            recv_grad_tensors_thread = threading.Thread(target=self.recv_grad_tensors)
            threads.append(recv_grad_tensors_thread)
            recv_grad_tensors_thread.start()
        
        if self.node_type == NodeTypes.LEAF or self.node_type == NodeTypes.STEM:
            backward_inner_pipe_monitor_thread = threading.Thread(target=self.backward_inner_pipe_monitor)
            threads.append(backward_inner_pipe_monitor_thread)
            backward_inner_pipe_monitor_thread.start()

            recv_forward_tensors_thread = threading.Thread(target=self.recv_forward_tensors)
            threads.append(recv_forward_tensors_thread)
            recv_forward_tensors_thread.start()

        self.start_server_flag.value = True

        for thread in threads:
            thread.join()

    def start_proc(self):
        serve_process = mp.Process(target=self.start, daemon=True)
        serve_process.start()

    def create_backward_payload(self, forward_pass_id=None, model_args=None):
        # grads = []
        if self.node_type == NodeTypes.LEAF:
            # if isinstance(model_args, torch.Tensor):
                # grads.append(model_args.grad.detach().to(torch.device('cpu')))
            grads = model_args.grad.detach().to(torch.device('cpu'))
            # else:
            #     for value in model_args:
            #         if value.requires_grad:
            #             grads.append(value.grad.detach().to(torch.device('cpu')))
        else:
            # if isinstance(self.input_tensors[forward_pass_id], torch.Tensor):
            #     grads.append(self.input_tensors[forward_pass_id].grad.detach().to(torch.device('cpu')))
            # else:
            #     for value in self.input_tensors[forward_pass_id]:
            #         if value.requires_grad:
            #             grads.append(value.grad.detach().to(torch.device('cpu')))
            grads = self.input_tensors[forward_pass_id].grad.detach().to(torch.device('cpu'))
        return (forward_pass_id, grads)

    def __getstate__(self):
        print('Get state in comm tcp')
        return dict(
            start_server_flag = self.start_server_flag,
            forward_lock = self.forward_lock,
            backward_lock = self.backward_lock,
            load_forward_buffer = self.load_forward_buffer,
            load_backward_buffer = self.load_backward_buffer,
            backend = self.backend,
            rank = self.rank,
            node_type = self.node_type,
            world_size = self.world_size,
            forward_inner_pipe = self.forward_inner_pipe,
            backward_inner_pipe = self.backward_inner_pipe,
            forward_input_shape = self.forward_input_shape,
            backward_input_shape = self.backward_input_shape
        )

    def __setstate__(self, state):
        print('Set state in comm tcp')
        self.backend = state['backend']
        self.rank = state['rank']
        self.node_type = state['node_type']
        self.world_size = state['world_size']
        self.forward_inner_pipe = state['forward_inner_pipe']
        self.backward_inner_pipe = state['backward_inner_pipe']
        self.load_forward_buffer = state['load_forward_buffer']
        self.load_backward_buffer = state['load_backward_buffer']
        self.forward_lock = state['forward_lock']
        self.backward_lock = state['backward_lock']
        self.forward_input_shape = state['forward_input_shape']
        self.backward_input_shape = state['backward_input_shape']
        self.start_server_flag = state['start_server_flag']

        