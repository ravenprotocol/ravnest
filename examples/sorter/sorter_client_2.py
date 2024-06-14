import torch
import time
from ravnest import Node, set_seed
import pickle

set_seed(42)

with open('examples/sorter/sorter_data/y_train.pkl', 'rb') as fout_y:   
    y_train = pickle.load(fout_y)

def sorter_criterion(outputs, targets):
    return torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1)

if __name__ == '__main__':

    node = Node(name = 'node_2',
                optimizer = torch.optim.Adam,
                criterion = sorter_criterion, # Custom defined Criterion
                labels = torch.tensor(y_train), 
                device=torch.device('cpu')
                )
 
    while True:
        time.sleep(0)