import torch
import time
from ravnest.node import Node

host = '0.0.0.0'
port = 8081
model = torch.jit.load('cnn/submod_1.pt')

if __name__ == '__main__':
    node = Node(name='n1', template_path='cnn/templates', submod_file='submod_1', local_host=host, local_port=port, model=model, optimizer=torch.optim.Adam, forward_target_host='0.0.0.0', forward_target_port=8082, backward_target_host='0.0.0.0', backward_target_port=8080)
    node.start()
    while True:
        time.sleep(1)
