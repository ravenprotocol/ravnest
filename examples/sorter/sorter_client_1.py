import torch
import time
from ravnest.node import Node
from ravnest.utils import load_node_json_configs

if __name__ == '__main__':
    
    node_name = 'node_1'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer=torch.optim.Adam
    
    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer, 
                device=torch.device('cpu'),
                **node_metadata
                )

    while True:
        time.sleep(0)
