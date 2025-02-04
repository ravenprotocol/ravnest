import os
import torch
import torch.distributed as dist
import threading
import time
from ..strings import *
from .utils import check_works_thread, check_work_thread

class Communication_Torch():
    def __init__(self, backend='gloo', 
                 input_tensors=None, forward_input_shapes=None, 
                 backward_input_shapes=None, device=None):

        self.backend = backend
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        if self.rank == 0:
            self.node_type == NodeTypes.ROOT
        elif self.rank == self.world_size - 1:
            self.node_type = NodeTypes.LEAF
        else:
            self.node_type = NodeTypes.STEM

        self.input_tensors = input_tensors
        self.device = device

        self.forward_input_shapes=forward_input_shapes
        self.backward_input_shapes=backward_input_shapes
        self.forward_send_work, self.backward_send_work = None, None

        self.prepare_process_groups()

        if self.node_type == NodeTypes.ROOT or self.node_type == NodeTypes.STEM:
            self.backward_work = None
            self.backward_ips = None
            self.start_backward_recv()

        if self.node_type == NodeTypes.LEAF or self.node_type == NodeTypes.STEM:
            self.forward_work = None
            self.forward_ips = None
            self.start_forward_recv()

    def start_forward_recv(self):
        self.forward_ips = [torch.zeros(forward_input_shape).to(self.device) for forward_input_shape in self.forward_input_shapes]
        works = [dist.broadcast(forward_ip, self.rank - 1, group=self.prv_grp_fwd, async_op=True) for forward_ip in self.forward_ips]
        self.forward_recv_works = check_works_thread(works, type='recv_fwd')

    def start_backward_recv(self):
        self.backward_ips = [torch.zeros(backward_input_shape).to(self.device) for backward_input_shape in self.backward_input_shapes]
        works = [dist.broadcast(backward_ip, self.rank + 1, group=self.nxt_grp_bwd, async_op=True) for backward_ip in self.backward_ips]
        self.backward_recv_works = check_works_thread(works, type='recv_backward')

    def trigger_forward_send(self, data):
        if self.forward_send_work is not None:
            while not self.forward_send_work.is_completed(): 
                time.sleep(0)

        work = dist.broadcast(data, self.rank, group=self.nxt_grp_fwd, async_op=True)
        self.forward_send_work = check_work_thread(work, type='send_fwd')

    def trigger_backward_send(self, data):
        if self.backward_send_work is not None:
            while not self.backward_send_work.is_completed():
                time.sleep(0)

        work = dist.broadcast(data, self.rank, group=self.prv_grp_bwd, async_op=True)
        self.backward_send_work = check_work_thread(work, type='send_backward')

    def forward_recv_works_done(self):
        ready_flag = True
        for fwd_recv_work in self.forward_recv_works:
            if not fwd_recv_work.is_completed():
                ready_flag = False
        return ready_flag

    def backward_recv_works_done(self):
        ready_flag = True
        for bwd_work in self.backward_recv_works:
            if not bwd_work.is_completed():
                ready_flag = False
        return ready_flag

    def prepare_process_groups(self):
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)
        self.proc_grps = {}
        for i in range(self.world_size - 1):
            self.proc_grps[i] = {}
            self.proc_grps[i]['forward'] = dist.new_group([i, i+1])
            self.proc_grps[i]['backward'] = dist.new_group([i, i+1])
            print('Proc_grp created for: ', i, i+1)

        if self.node_type == NodeTypes.ROOT:
            self.nxt_grp_fwd = self.proc_grps[self.rank]['forward']
            self.nxt_grp_bwd = self.proc_grps[self.rank]['backward']
            
        elif self.node_type == NodeTypes.STEM:
            self.nxt_grp_fwd = self.proc_grps[self.rank]['forward']
            self.nxt_grp_bwd = self.proc_grps[self.rank]['backward']                
            self.prv_grp_fwd = self.proc_grps[self.rank - 1]['forward']
            self.prv_grp_bwd = self.proc_grps[self.rank - 1]['backward']                
        else:
            self.prv_grp_fwd = self.proc_grps[self.rank - 1]['forward']
            self.prv_grp_bwd = self.proc_grps[self.rank - 1]['backward']

    def create_backward_payload(self, forward_pass_id=None, model_args=None):
        # grads = []
        if self.node_type == NodeTypes.LEAF:
            # if isinstance(model_args, torch.Tensor):
                # grads.append(model_args.grad.detach().to(torch.device('cpu')))
            # print(len(model_args))
            grads = [model_arg.grad.detach() for model_arg in model_args]#.to(torch.device('cpu'))
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
            grads = [input_tensor.grad.detach() for input_tensor in self.input_tensors[forward_pass_id]]#.to(torch.device('cpu'))
        return grads #(forward_pass_id, grads)
        