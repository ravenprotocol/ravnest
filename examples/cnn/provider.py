import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import datasets
from sklearn.model_selection import train_test_split
from ravnest import Node, set_seed
from ravnest.trainer import BaseTrainerFullAsync#Trainer

set_seed(42)

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def preprocess_dataset():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1, 1, 8, 8)).astype(np.float32)
    X_test = X_test.reshape((-1, 1, 8, 8)).astype(np.float32)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(list(zip(X_train,torch.tensor(y_train, dtype=torch.float32))), generator=generator, shuffle=True, batch_size=64)
    val_loader = DataLoader(list(zip(X_test,torch.tensor(y_test, dtype=torch.float32))), shuffle=False, batch_size=64)

    return train_loader, val_loader

def loss_fn(preds, targets):
    return torch.nn.functional.mse_loss(preds, targets)

def accuracy_fn(preds, y_test):
    _, y_pred_tags = torch.max(preds, dim=1)
    #for cnn
    y_test = torch.argmax(y_test, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    val_acc = correct_pred.sum() / len(y_test)
    val_acc = torch.round(val_acc * 100)
    return val_acc

# train_loader, val_loader = preprocess_dataset()

if __name__ == '__main__':

    train_loader, val_loader = preprocess_dataset()

    node = Node(name = 'node_0', 
                optimizer = torch.optim.Adam,
                device=torch.device('cpu'),
                criterion = loss_fn,
                labels = train_loader,
                test_labels=val_loader,
                # reduce_factor=4,
                average_optim=True
                )

    trainer = BaseTrainerFullAsync(node=node,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      val_freq=64,
                      epochs=100,
                      batch_size=64,
                    #   save=True,
                      loss_fn=loss_fn,
                      accuracy_fn=accuracy_fn
                    )

    trainer.train()

    trainer.evaluate()