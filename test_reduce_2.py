from node import Node
import numpy as np
import time

host = '0.0.0.0'
port = 8081

data = np.array([[1,2,3],[4,5,6],[7,8,9]])
ring_ids={0:0}
rank=1
ring_size = 2
data_dict = {}
for i in range(len(data)):
    data_dict[i] = data[i]
param_addresses = {'0.0.0.0:8080':0}

if __name__ == '__main__':
    node = Node(name='n1', local_host=host, local_port=port, ring_ids=ring_ids, ring_size=ring_size, rank=rank, data_dict=data_dict, param_addresses=param_addresses) #forward_target_host='0.0.0.0', forward_target_port=8081)
    node.start()
    time.sleep(3)
    node.parallel_ring_reduce()
    while True:
        time.sleep(1)
