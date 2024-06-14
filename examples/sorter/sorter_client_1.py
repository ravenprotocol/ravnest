import torch
import time
from ravnest import Node, set_seed

set_seed(42)

if __name__ == '__main__':
        
    node = Node(name = 'node_1',
                optimizer = torch.optim.Adam, 
                device=torch.device('cpu')
                )

    while True:
        time.sleep(0)
