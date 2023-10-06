import torch
import time
from node import Node
from utils import current_model_params_clone

host = '0.0.0.0'
port = 8081
model = torch.jit.load('submod_1.pt')

# optimizer = torch.optim.Adam(current_model_params_clone(model))

if __name__ == '__main__':
    node = Node(name='n1', local_host=host, local_port=port, model=model, optimizer=torch.optim.Adam, forward_target_host='0.0.0.0', forward_target_port=8082, backward_target_host='0.0.0.0', backward_target_port=8080)
    node.start()
    while True:
        time.sleep(1)
