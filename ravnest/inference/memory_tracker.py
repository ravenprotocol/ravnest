import torch
import psutil
from ..strings import *

class MemoryTracker():
    def __init__(self, comm_session=None, device=None):
        self.comm_session = comm_session
        self.node_type = self.comm_session.node_type
        self.device = device
        if self.node_type == NodeTypes.ROOT:
            self.memory_metrics = {
                'gpu_usage':[torch.tensor(0, dtype=torch.float16).to(self.device)]*self.comm_session.world_size,
                'cpu_usage':[torch.tensor(0, dtype=torch.float16).to(self.device)]*self.comm_session.world_size,
            }

    def get_gpu_usage(self):
        total_memory = torch.cuda.get_device_properties(self.device).total_memory            
        allocated_memory = torch.cuda.memory_allocated(self.device)
        gpu_percent = round((allocated_memory / total_memory) * 100, 1)
        return gpu_percent
    
    def get_cpu_usage(self):
        cpu_percent = round(psutil.virtual_memory().percent, 1)
        return cpu_percent

    def update_metrics(self):
        if self.node_type == NodeTypes.ROOT:
            gpu_mem_list, cpu_mem_list = self.memory_metrics['gpu_usage'], self.memory_metrics['cpu_usage']
        else:
            gpu_mem_list, cpu_mem_list = None, None
        
        curr_gpu_allocated = torch.tensor(self.get_gpu_usage(), dtype=torch.float16).to(self.device)
        self.comm_session.gather_at_root(curr_gpu_allocated, gpu_mem_list)
        
        curr_cpu_allocated = torch.tensor(self.get_cpu_usage(), dtype=torch.float16).to(self.device)
        self.comm_session.gather_at_root(curr_cpu_allocated, cpu_mem_list)

    def get_metrics(self):
        if self.node_type == NodeTypes.ROOT:
            gpu_memories = [round(mem.item(),1) for mem in self.memory_metrics['gpu_usage']]
            cpu_memories = [round(mem.item(),1) for mem in self.memory_metrics['cpu_usage']]
            return gpu_memories, cpu_memories
        return None, None