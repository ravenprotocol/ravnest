import torch
from torch.utils.data import DataLoader
from ravnest import Node, Trainer, set_seed
import pickle

set_seed(42)

with open('examples/sorter/sorter_data/X_train.pkl', 'rb') as fout_X:
    X_train = pickle.load(fout_X)

with open('examples/sorter/sorter_data/y_train.pkl', 'rb') as fout_y:   
    y_train = pickle.load(fout_y)

def sorter_criterion(outputs, targets):
    return torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets[1].view(-1), ignore_index=-1)

train_loader = DataLoader(list(zip(X_train,y_train)), shuffle=False, batch_size=64)

if __name__ == '__main__':

    node = Node(name = 'node_0',
                optimizer = torch.optim.Adam,
                criterion = sorter_criterion, # Custom defined Criterion
                labels = torch.tensor(y_train), 
                device=torch.device('cpu')
                )

    trainer = Trainer(node=node,
                      train_loader=train_loader,
                      save=True,
                      epochs=1,
                      batch_size=64
                      )

    trainer.train()
