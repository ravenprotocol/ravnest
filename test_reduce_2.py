from node import Node
import numpy as np
import time

host = '0.0.0.0'
port = 8081

data = np.array([[7,8,9]]) #[1,2,3],[4,5,6]
# data = np.random.random(size=(3,3,4))
# param_names = ['a','b','c']

ring_ids={1:2}
rank=1
ring_size = 2
data_dict = {}
data_dict[2] = data[0]
# for i in range(len(data)):
#     data_dict[i] = data[i]
    # data_dict[param_names[i]] = data[i]

param_addresses = {'0.0.0.0:8080':2}

if __name__ == '__main__':
    node = Node(name='n1', local_host=host, local_port=port, ring_ids=ring_ids, ring_size=ring_size, rank=rank, data_dict=data_dict, param_addresses=param_addresses) #forward_target_host='0.0.0.0', forward_target_port=8081)
    node.start()
    time.sleep(3)
    node.parallel_ring_reduce()
    while True:
        time.sleep(1)
