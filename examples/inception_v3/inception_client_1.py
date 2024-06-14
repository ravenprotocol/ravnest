import torch
import time
from ravnest import Node, set_seed

set_seed(42)

if __name__ == '__main__':
    
    node = Node(name = 'node_1',
                optimizer = torch.optim.SGD, 
                optimizer_params = {'lr':0.01, 'momentum':0.9, 'weight_decay':0.0005},
                device=torch.device('cpu')
                )

    while True:
        time.sleep(0)
