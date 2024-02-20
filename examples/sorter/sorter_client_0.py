import torch
from torch.utils.data import DataLoader
from ravnest.node import Node
from ravnest.trainer import Trainer
from ravnest.utils import load_node_json_configs
import time
import numpy as np
import random
import pickle

np.random.seed(42)
random.seed(42)

with open('examples/sorter/sorter_data/X_train.pkl', 'rb') as fout_X:
    X_train = pickle.load(fout_X)

with open('examples/sorter/sorter_data/y_train.pkl', 'rb') as fout_y:   
    y_train = pickle.load(fout_y)

train_loader = DataLoader(list(zip(X_train,y_train)), shuffle=False, batch_size=64)

if __name__ == '__main__':

    node_name = 'node_0'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer=torch.optim.Adam
    
    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer,
                **node_metadata
                )

    trainer = Trainer(node=node,
                      train_loader=train_loader,
                      save=True,
                      epochs=1,
                      batch_size=64
                      )

    t1 = time.time()

    trainer.train()

    print('Training Done!: ', time.time() - t1)
