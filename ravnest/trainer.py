import torch
import numpy as np
import time
from .strings import *

class Trainer():
    """
    A Trainer class for training machine learning models with support for custom learning rate schedulers, 
    training and validation data loaders, and configurable training parameters. This class can be extended to create custom trainers as per your training requirements.

    :param node: A ``ravnest.node`` object that contains the optimizer for training.
    :type node: object
    :param lr_scheduler: Learning rate scheduler for the optimizer, defaults to None.
    :type lr_scheduler: object, optional
    :param lr_scheduler_params: Parameters for the learning rate scheduler, defaults to None.
    :type lr_scheduler_params: dict, optional
    :param train_loader: DataLoader for the training dataset.
    :type train_loader: DataLoader
    :param val_loader: DataLoader for the validation dataset, defaults to None.
    :type val_loader: DataLoader, optional
    :param val_freq: Frequency of validation checks during training, defaults to 1.
    :type val_freq: int, optional
    :param save: Flag to indicate whether to save the submodel (in folder 'trained_submodels') after training, defaults to False.
    :type save: bool, optional
    :param epochs: Number of epochs to train the model, defaults to 1.
    :type epochs: int, optional
    :param batch_size: Size of the training batches, defaults to 64.
    :type batch_size: int, optional
    :param step_size: Step size for gradient updates, defaults to 1.
    :type step_size: int, optional
    :param inputs_dtype: Data type for the input tensors, defaults to None.
    :type inputs_dtype: torch.dtype, optional
    :param n_forwards: Counter for the number of forward passes, initialized to 0.
    :type n_forwards: int
    """

    def __init__(self, node=None, lr_scheduler=None, lr_scheduler_params={}, train_loader=None, val_loader=None, val_freq=1, save=False, epochs=1, batch_size=64, step_size=1, inputs_dtype=None):
        self.node = node
        if self.node.node_type == NodeTypes.STEM or self.node.node_type == NodeTypes.LEAF:
            return
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

    def prelim_checks(self):
        if self.node.node_type == NodeTypes.STEM or self.node.node_type == NodeTypes.LEAF:
            while True:
                time.sleep(0)

    def train(self):
        """Train the model using the specified training and validation data loaders.

        Iterates over the training data for the specified number of epochs,
        performing forward computations, updating parameters, and optionally
        evaluating on validation data.
        """
        self.prelim_checks()
        t1 = time.time()
        self.n_forwards = 0
        for epoch in range(self.epochs):
            for X_train, y_train in self.train_loader:
                if torch.is_tensor(X_train):
                    if self.inputs_dtype is not None:
                        if X_train.dtype == self.inputs_dtype:
                            self.node.forward_compute(tensors=X_train)
                        else:
                            self.node.forward_compute(tensors=torch.tensor(X_train.numpy(), dtype=self.inputs_dtype))
                    else:
                        self.node.forward_compute(tensors=X_train)
                else:
                    if X_train.dtype == self.inputs_dtype:
                        self.node.forward_compute(tensors=torch.tensor(X_train.numpy(), dtype=self.inputs_dtype))
                    else:
                        self.node.forward_compute(tensors=torch.tensor(X_train.numpy()))
                self.n_forwards += 1                
            
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
        
    def pred(self, data):
        """Perform prediction on sample test data using the trained model.

        :param data: Sample data for prediction, can be numpy array or torch tensor
        :type data: np.ndarray or torch.Tensor
        :return: Prediction result
        :rtype: torch.Tensor
        """
        if self.node.node_type == NodeTypes.STEM or self.node.node_type == NodeTypes.LEAF:
            return
        if isinstance(data, np.ndarray):
            pred = self.node.no_grad_forward_compute(tensors=torch.tensor(data, dtype=torch.float32), output_type='accuracy')
        else:
            pred = self.node.no_grad_forward_compute(tensors=torch.tensor(data.numpy(), dtype=torch.float32), output_type='accuracy')
        return pred
    
    def evaluate(self):
        """Evaluate the trained model on the validation data.

        Performs inference on validation data and computes validation accuracy
        using the trained model.
        """
        if self.node.node_type == NodeTypes.STEM or self.node.node_type == NodeTypes.LEAF:
            return
        for X_test, y_test in self.val_loader:
            self.node.no_grad_forward_compute(tensors=torch.tensor(X_test.numpy(), dtype=self.inputs_dtype), output_type='val_accuracy')