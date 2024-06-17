import torch
import numpy as np
import time

class Trainer():
    def __init__(self, node=None, lr_scheduler=None, lr_scheduler_params={}, train_loader=None, val_loader=None, val_freq=1, save=False, epochs=1, batch_size=64, step_size=1, inputs_dtype=None):
        """A Trainer class for training machine learning models with support for custom learning rate schedulers, training and validation data loaders, and configurable training parameters.

        Attributes:
        -----------
        node : object
            An object that contains the model and optimizer for training.
        lr_scheduler : object, optional
            Learning rate scheduler for the optimizer.
        lr_scheduler_params : dict, optional
            Parameters for the learning rate scheduler.
        train_loader : DataLoader
            DataLoader for the training dataset.
        val_loader : DataLoader, optional
            DataLoader for the validation dataset.
        val_freq : int, optional
            Frequency of validation checks during training, default is 1.
        save : bool, optional
            Flag to indicate whether to save the model after training, default is False.
        epochs : int, optional
            Number of epochs to train the model, default is 1.
        batch_size : int, optional
            Size of the training batches, default is 64.
        step_size : int, optional
            Step size for gradient updates, default is 1.
        inputs_dtype : torch.dtype, optional
            Data type for the input tensors, default is None.
        n_forwards : int
            Counter for the number of forward passes, initialized to 0.

        Methods:
        --------
        train():
            Trains the model for a specified number of epochs.
        pred(input):
            Makes a prediction using the trained model on the given input.
        evaluate():
            Evaluates the model using the validation dataset.
        """
        self.node = node
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_freq = val_freq
        self.save = save
        self.epochs = epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.n_forwards = 0
        self.inputs_dtype = inputs_dtype
        self.lr_scheduler=None
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.node.optimizer, **lr_scheduler_params)

    def train(self):
        t1 = time.time()
        self.n_forwards = 0
        for epoch in range(self.epochs):
            data_id = 0
            for X_train, y_train in self.train_loader:
                if torch.is_tensor(X_train):
                    if self.inputs_dtype is not None:
                        if X_train.dtype == self.inputs_dtype:
                            self.node.forward_compute(data_id=data_id, tensors=X_train)
                        else:
                            self.node.forward_compute(data_id=data_id, tensors=torch.tensor(X_train.numpy(), dtype=self.inputs_dtype))
                    else:
                        self.node.forward_compute(data_id=data_id, tensors=X_train)
                else:
                    if X_train.dtype == self.inputs_dtype:
                        self.node.forward_compute(data_id=data_id, tensors=torch.tensor(X_train.numpy(), dtype=self.inputs_dtype))
                    else:
                        self.node.forward_compute(data_id=data_id, tensors=torch.tensor(X_train.numpy()))
                self.n_forwards += 1                
                data_id += (self.batch_size // self.step_size)
            
            if self.val_loader is not None: 
                # if self.n_forwards % self.val_freq == 0:
                for X_test, y_test in self.val_loader:
                    self.node.no_grad_forward_compute(tensors=torch.tensor(X_test.numpy(), dtype=self.inputs_dtype), output_type='val_accuracy')

            self.node.wait_for_backwards()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            print('Epoch: ', epoch)
        
        print('Training Done!: ', time.time() - t1, ' seconds')

        if self.save:
            self.node.trigger_save_submodel()
        
    def pred(self, input):
        if isinstance(input, np.ndarray):
            pred = self.node.no_grad_forward_compute(tensors=torch.tensor(input, dtype=torch.float32), output_type='accuracy')
        else:
            pred = self.node.no_grad_forward_compute(tensors=torch.tensor(input.numpy(), dtype=torch.float32), output_type='accuracy')
        return pred
    
    def evaluate(self):
        for X_test, y_test in self.val_loader:
            self.node.no_grad_forward_compute(tensors=torch.tensor(X_test.numpy(), dtype=self.inputs_dtype), output_type='val_accuracy')