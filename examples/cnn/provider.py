import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
from ravnest import Node, Trainer, set_seed

set_seed(42)

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

train_loader = DataLoader(list(zip(X,torch.tensor(y, dtype=torch.float32))), shuffle=False, batch_size=64)
val_loader = DataLoader(list(zip(X_test,torch.tensor(y_test, dtype=torch.float32))), shuffle=False, batch_size=64)

if __name__ == '__main__':

    node = Node(name = 'node_0', 
                optimizer = torch.optim.Adam,
                device=torch.device('cpu'),
                criterion = torch.nn.functional.mse_loss, 
                labels = train_loader, 
                test_labels=val_loader
                )

    trainer = Trainer(node=node,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      val_freq=64,
                      epochs=100,
                      batch_size=64,
                      inputs_dtype=torch.float32)

    trainer.train()

    trainer.evaluate()
