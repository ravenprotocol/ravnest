import torch
import numpy as np
import random
import time
from sklearn import datasets
from ravnest.node import Node
from sklearn.model_selection import train_test_split
import pickle

np.random.seed(42)
random.seed(42)

with open('sorter/sorter_data/X_train.pkl', 'rb') as fout_X:
    X_train = pickle.load(fout_X)

with open('sorter/sorter_data/y_train.pkl', 'rb') as fout_y:   
    y_train = pickle.load(fout_y)

host = '0.0.0.0'
port = 8082
model = torch.jit.load('sorter/submod_2.pt')

if __name__ == '__main__':
    node = Node(name='n2', template_path='sorter/templates', submod_file='submod_2', local_host=host, local_port=port, labels=torch.tensor(y_train), model=model, optimizer=torch.optim.Adam, backward_target_host='0.0.0.0', backward_target_port=8081)

    node.start()
    while True:
        time.sleep(1)
