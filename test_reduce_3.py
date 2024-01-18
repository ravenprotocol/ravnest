from ravnest.node import Node
import numpy as np
import time
import torch

host = '0.0.0.0'
port = 8082


if __name__ == '__main__':

    model = torch.jit.load('node_data/cluster_0/192.128.30.92:8082/submod.pt')
    param_addresses = {'192.128.30.92:8080': 'conv2d_1.weight'}
    ring_ids = {0: 'conv2d_1.weight'}
    data_dict = model.state_dict()
    rank = 0
    ring_size = 2


    node = Node(name='n1', model=model, local_host=host, local_port=port, ring_ids=ring_ids, ring_size=ring_size, rank=rank, data_dict=data_dict, param_addresses=param_addresses) #forward_target_host='0.0.0.0', forward_target_port=8081)
    node.start()
    time.sleep(3)
    node.parallel_ring_reduce()
    while True:
        time.sleep(1)
