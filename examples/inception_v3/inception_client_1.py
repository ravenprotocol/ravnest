import torch
import time
from ravnest.node import Node
from ravnest.utils import load_node_json_configs

if __name__ == '__main__':
    
    node_name = 'node_1'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer = torch.optim.SGD
    optimizer_params = {'lr':0.01, 'momentum':0.9, 'weight_decay':0.0005}
    
    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer, 
                optimizer_params = optimizer_params,
                device=torch.device('cuda'),
                **node_metadata
                )

    while True:
        time.sleep(0)
