import torch
import numpy as np
import time
from sklearn import datasets
from node import Node
from sklearn.model_selection import train_test_split
import _pickle as cPickle

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

host = '0.0.0.0'
port = 8082
model = torch.jit.load('submod_2.pt')
# optimizer = torch.optim.Adam(model.parameters())

if __name__ == '__main__':
    node = Node(name='n2', local_host=host, local_port=port, labels=torch.tensor(y, dtype=torch.float32), test_labels = y_test, model=model, optimizer=torch.optim.Adam, backward_target_host='0.0.0.0', backward_target_port=8081)

    node.start()
    while True:
        time.sleep(1)
