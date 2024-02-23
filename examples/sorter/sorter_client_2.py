import torch
import numpy as np
import random
import time
from ravnest.node import Node
from ravnest.utils import load_node_json_configs
import pickle

np.random.seed(42)
random.seed(42)

with open('examples/sorter/sorter_data/y_train.pkl', 'rb') as fout_y:   
    y_train = pickle.load(fout_y)

def sorter_criterion(outputs, targets):
    return torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1)

if __name__ == '__main__':

    node_name = 'node_2'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer=torch.optim.Adam                
    criterion = sorter_criterion   # Custom defined Criterion

    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer,
                criterion = criterion, 
                labels = torch.tensor(y_train), 
                device=torch.device('cpu'),
                **node_metadata
                )
 
    while True:
        time.sleep(1)