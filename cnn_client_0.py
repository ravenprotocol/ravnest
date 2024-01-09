import torch
import numpy as np
import time
from sklearn import datasets
from ravnest.node import Node
from sklearn.model_selection import train_test_split
import time

def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def get_dataset():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))

    return X_train, X_test, y_train, y_test

X, X_test, y, y_test = get_dataset()
model = torch.jit.load('cnn/submod_0.pt')


host = '0.0.0.0'
port = 8080

if __name__ == '__main__':
    node = Node(name='n0', template_path='cnn/templates', submod_file='submod_0', local_host=host, local_port=port, model=model, optimizer=torch.optim.Adam, forward_target_host='0.0.0.0', forward_target_port=8081)
    node.start()
    batch_size = 64
    epochs = 100
    tensor_id = 0
    t1 = time.time()
    n_forwards = 0


    for epoch in range(epochs):
        data_id = 0
        for X_train, y_train in batch_iterator(X,y,batch_size=batch_size):
            # print('\nPassing batch')
            # output = model(torch.tensor(X_train, dtype=torch.float32))
            node.forward_compute(data_id=data_id, tensors=torch.tensor(X_train, dtype=torch.float32))
            n_forwards += 1
            # print('Data id: ', data_id)
            
            data_id += batch_size
            # break
            # if data_id>256:
            #     break
        print('Epoch: ', epoch)
        print('n_forward: ', n_forwards, '  node.n_backward: ', node.n_backwards)
    
    while node.n_backwards < n_forwards: #node.is_training: #
        time.sleep(1)
    
    print('Training Done!: ', time.time() - t1)

    
    pred = node.no_grad_forward_compute(tensors=torch.tensor(X_test, dtype=torch.float32), output_type='accuracy')

    


