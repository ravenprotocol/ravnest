import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
from ravnest.node import Node
from ravnest.trainer import Trainer
from ravnest.utils import load_node_json_configs


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

train_loader = DataLoader(list(zip(X,y)), shuffle=False, batch_size=64)

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
                      epochs=100,
                      batch_size=64,
                      inputs_dtype=torch.float32)

    trainer.train()

    trainer.pred(input=X_test)
