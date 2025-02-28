import os
import torch
import torch.distributed as dist
import threading
import time
from ..strings import *
from .utils import check_works_thread, check_work_thread

class Communication_Torch():
    def __init__(self, backend='gloo', mode=NodeModes.TRAIN,
                 dist_timeout=None,
                 input_tensors=None, forward_input_shapes=None,
                 feedback_shape=None, backward_input_shapes=None, 
                 dtype=None, device=None):

        self.backend = backend
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.node_type = None
        self.mode = mode
        self.dtype = dtype
        self.dist_timeout = dist_timeout
        if self.rank == 0:
            self.node_type = NodeTypes.ROOT
        elif self.rank == self.world_size - 1:
            self.node_type = NodeTypes.LEAF
        else:
            self.node_type = NodeTypes.STEM

        self.input_tensors = input_tensors
        self.device = device

        self.forward_input_shapes=forward_input_shapes
        self.backward_input_shapes=backward_input_shapes
        self.feedback_shape = feedback_shape
        self.forward_send_work, self.backward_send_work, self.feedback_send_work = None, None, None

        self.nxt_grp_fwd_stream, self.nxt_grp_bwd_stream = None, None
        self.prv_grp_fwd_stream, self.prv_grp_bwd_stream = None, None
        self.feedback_stream = None

        self.prepare_process_groups()
        print('prepare_process_groups done')
        
        if self.node_type == NodeTypes.ROOT or self.node_type == NodeTypes.STEM:
            self.backward_work = None
            self.backward_ips = None
            
        if self.node_type == NodeTypes.LEAF or self.node_type == NodeTypes.STEM:
            self.forward_work = None
            self.forward_ips = None
            self.is_receiving_fwd = False
            
    def start_forward_recv(self):
        '''
        TODO: Remove work.wait() in thread for nccl
        '''

        self.forward_ips = [torch.zeros(forward_input_shape, dtype=self.dtype).to(self.device) for forward_input_shape in self.forward_input_shapes]
        # works = [dist.broadcast(forward_ip, self.rank - 1, group=self.prv_grp_fwd, async_op=True) for forward_ip in self.forward_ips]
        works = [dist.irecv(forward_ip, self.rank - 1, group=self.prv_grp_fwd) for forward_ip in self.forward_ips]
        self.forward_recv_works = check_works_thread(works, stream=self.prv_grp_fwd_stream, type='recv_fwd')

    def start_backward_recv(self):
        self.backward_ips = [torch.zeros(backward_input_shape, dtype=self.dtype).to(self.device) for backward_input_shape in self.backward_input_shapes]
        # works = [dist.broadcast(backward_ip, self.rank + 1, group=self.nxt_grp_bwd, async_op=True) for backward_ip in self.backward_ips]
        works = [dist.irecv(backward_ip, self.rank + 1, group=self.nxt_grp_bwd) for backward_ip in self.backward_ips]
        self.backward_recv_works = check_works_thread(works, stream=self.nxt_grp_bwd_stream, type='recv_backward')
    
    def start_feedback_recv(self):
        self.feedback_ip = torch.zeros(self.feedback_shape, dtype=torch.int64).to(self.device)
        # work = dist.irecv(self.feedback_ip, self.world_size - 1, group=self.feedback_grp)
        work = dist.broadcast(self.feedback_ip, self.world_size - 1, group=self.feedback_grp, async_op=True)
        self.feedback_recv_work = check_work_thread(work, stream=self.feedback_stream, type='recv_feedback')

    def trigger_forward_send(self, data):
        if self.forward_send_work is not None:
            while not self.forward_send_work.is_completed(): 
                time.sleep(0)

        # work = dist.broadcast(data, self.rank, group=self.nxt_grp_fwd, async_op=True)
        work = dist.isend(data, self.rank + 1, group=self.nxt_grp_fwd)
        self.forward_send_work = check_work_thread(work, stream=self.nxt_grp_fwd_stream, type='send_fwd')

    def trigger_backward_send(self, data):
        if self.backward_send_work is not None:
            while not self.backward_send_work.is_completed():
                time.sleep(0)

        # work = dist.broadcast(data, self.rank, group=self.prv_grp_bwd, async_op=True)
        work = dist.isend(data, self.rank - 1, group=self.prv_grp_bwd)
        self.backward_send_work = check_work_thread(work, stream=self.prv_grp_bwd_stream, type='send_backward')

    def trigger_feedback_send(self, data):
        if self.feedback_send_work is not None:
            while not self.feedback_send_work.is_completed():
                time.sleep(0)

        work = dist.broadcast(data, self.rank, group=self.feedback_grp, async_op=True)
        # work = dist.isend(data, 0, group=self.feedback_grp)
        self.feedback_send_work = check_work_thread(work, stream=self.feedback_stream, type='send_feedback')

    def broadcast_metadata(self, data):
        w = dist.broadcast(data, 0, group=self.metadata_grp, async_op=True)
        w.wait()

    def broadcast_metadata_objects(self, data):
        dist.broadcast_object_list(data, 0, group=self.metadata_grp)
    
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

    def feedback_recv_work_done(self):
        return self.feedback_recv_work.is_completed()

    def prepare_process_groups(self):
        dist.init_process_group(backend=self.backend, rank=self.rank, timeout=self.dist_timeout, world_size=self.world_size)
        self.proc_grps = {}
        for i in range(self.world_size - 1):
            self.proc_grps[i] = {}
            self.proc_grps[i]['forward'] = dist.new_group([i, i+1], timeout=self.dist_timeout)
            if self.mode == NodeModes.TRAIN:
                self.proc_grps[i]['backward'] = dist.new_group([i, i+1], timeout=self.dist_timeout)
            print('Proc_grp created for: ', i, i+1)
        
        if self.mode == NodeModes.INFERENCE:
            self.metadata_grp = dist.new_group([i for i in range(self.world_size)], timeout=self.dist_timeout)
            self.feedback_grp = dist.new_group([i for i in range(self.world_size)], timeout=self.dist_timeout)

        if self.node_type == NodeTypes.ROOT:
            self.nxt_grp_fwd = self.proc_grps[self.rank]['forward']
            if self.mode == NodeModes.TRAIN:
                self.nxt_grp_bwd = self.proc_grps[self.rank]['backward']
            
        elif self.node_type == NodeTypes.STEM:
            self.nxt_grp_fwd = self.proc_grps[self.rank]['forward']
            self.prv_grp_fwd = self.proc_grps[self.rank - 1]['forward']

            if self.mode == NodeModes.TRAIN: 
                self.nxt_grp_bwd = self.proc_grps[self.rank]['backward']              
                self.prv_grp_bwd = self.proc_grps[self.rank - 1]['backward']
        else:
            self.prv_grp_fwd = self.proc_grps[self.rank - 1]['forward']

            if self.mode == NodeModes.TRAIN:
                self.prv_grp_bwd = self.proc_grps[self.rank - 1]['backward']
        
        if self.backend == 'nccl':
            self.create_cuda_streams()
            self.nccl_warmup_groups()
        
        print('Group warmups done')

    def nccl_warmup_groups(self):
        if self.node_type == NodeTypes.ROOT:
            w_send = dist.isend(torch.tensor([1]).to(self.device), self.rank + 1, group=self.nxt_grp_fwd)
            if self.mode == NodeModes.TRAIN:
                w_recv = dist.irecv(torch.tensor([0]).to(self.device), self.rank + 1, group=self.nxt_grp_bwd)
            elif self.mode == NodeModes.INFERENCE:
                # w_recv = dist.irecv(torch.tensor([0]).to(self.device), self.world_size - 1, group=self.feedback_grp)
                w_recv = dist.broadcast(torch.tensor([0]).to(self.device), self.world_size - 1, group=self.feedback_grp, async_op=True)
                w_send_metadata = dist.broadcast(torch.tensor([0]).to(self.device), self.rank, group=self.metadata_grp, async_op=True)
            w_send.wait()
            w_recv.wait()
            if self.mode == NodeModes.INFERENCE:
                w_send_metadata.wait()

        elif self.node_type == NodeTypes.STEM:
            w_recv = dist.irecv(torch.tensor([0]).to(self.device), self.rank - 1, group=self.prv_grp_fwd)
            w_send = dist.isend(torch.tensor([1]).to(self.device), self.rank + 1, group=self.nxt_grp_fwd)
            w_recv.wait()
            w_send.wait()

            if self.mode == NodeModes.TRAIN:
                w_recv = dist.irecv(torch.tensor([0]).to(self.device), self.rank + 1, group=self.nxt_grp_bwd)
                w_send = dist.isend(torch.tensor([1]).to(self.device), self.rank - 1, group=self.prv_grp_bwd)
                w_recv.wait()
                w_send.wait()
            elif self.mode == NodeModes.INFERENCE:
                w_recv = dist.broadcast(torch.tensor([0]).to(self.device), self.world_size - 1, group=self.feedback_grp, async_op=True)
                w_recv.wait()
                w_recv = dist.broadcast(torch.tensor([0]).to(self.device), 0, group=self.metadata_grp, async_op=True)
                w_recv.wait()
        else:
            w_recv = dist.irecv(torch.tensor([0]).to(self.device), self.rank - 1, group=self.prv_grp_fwd)
            if self.mode == NodeModes.TRAIN:
                w_send = dist.isend(torch.tensor([1]).to(self.device), self.rank - 1, group=self.prv_grp_bwd)
            elif self.mode == NodeModes.INFERENCE:
                # w_send = dist.isend(torch.tensor([1]).to(self.device), 0, group=self.feedback_grp)
                w_send = dist.broadcast(torch.tensor([1]).to(self.device), self.rank, group=self.feedback_grp, async_op=True)
                w_recv_metadata = dist.broadcast(torch.tensor([0]).to(self.device), 0, group=self.metadata_grp, async_op=True)
            w_recv.wait()
            w_send.wait()
            if self.mode == NodeModes.INFERENCE:
                w_recv_metadata.wait()

    def create_cuda_streams(self):
        if self.node_type == NodeTypes.ROOT:
            self.nxt_grp_fwd_stream = torch.cuda.Stream()
            if self.mode == NodeModes.TRAIN:
                self.nxt_grp_bwd_stream = torch.cuda.Stream()
            elif self.mode == NodeModes.INFERENCE:
                self.feedback_stream = torch.cuda.Stream()

        elif self.node_type == NodeTypes.STEM:
            self.nxt_grp_fwd_stream = torch.cuda.Stream()
            self.prv_grp_fwd_stream = torch.cuda.Stream()
            if self.mode == NodeModes.TRAIN:
                self.nxt_grp_bwd_stream = torch.cuda.Stream()
                self.prv_grp_bwd_stream = torch.cuda.Stream()
            elif self.mode == NodeModes.INFERENCE:
                self.feedback_stream = torch.cuda.Stream()
        
        else:
            self.prv_grp_fwd_stream = torch.cuda.Stream()
            if self.mode == NodeModes.TRAIN:
                self.prv_grp_bwd_stream = torch.cuda.Stream()
            elif self.mode == NodeModes.INFERENCE:
                self.feedback_stream = torch.cuda.Stream()
    
    def create_backward_payload(self, forward_pass_id=None, model_args=None):
        if self.node_type == NodeTypes.LEAF:
            grads = [model_arg.grad.detach() for model_arg in model_args]
        else:
            grads = [input_tensor.grad.detach() for input_tensor in self.input_tensors[forward_pass_id]]#.to(torch.device('cpu'))
        return grads
        