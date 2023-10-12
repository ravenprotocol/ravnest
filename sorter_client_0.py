import torch
from node import Node
import time
import numpy as np
import random
import pickle

np.random.seed(42)
random.seed(42)

def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

with open('sorter/sorter_data/X_train.pkl', 'rb') as fout_X:
    X_train = pickle.load(fout_X)

with open('sorter/sorter_data/y_train.pkl', 'rb') as fout_y:   
    y_train = pickle.load(fout_y)

# print('X_train: ', X_train[:5])
# print('y_train: ', y_train[:5])


model = torch.jit.load('sorter/submod_0.pt')

# print('Model: ', model)

host = '0.0.0.0'
port = 8080

if __name__ == '__main__':
    node = Node(name='n0', template_path='sorter/templates', submod_file='submod_0', local_host=host, local_port=port, model=model, optimizer=torch.optim.Adam, forward_target_host='0.0.0.0', forward_target_port=8081)
    node.start()
    batch_size = 64
    epochs = 10
    t1 = time.time()
    n_forwards = 0

    for epoch in range(epochs):
        data_id = 0
        for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size=batch_size):
            node.forward_compute(data_id=data_id, tensors=torch.tensor(X_batch))#, dtype=torch.float32))
            n_forwards += 1
            data_id += batch_size

        print('Epoch: ', epoch)
        print('n_forward: ', n_forwards, '  node.n_backward: ', node.n_backwards)
                

    while node.n_backwards < n_forwards: #node.is_training: #
        time.sleep(1)
    
    print('Training Done!: ', time.time() - t1)

    node.trigger_save_submodel()


