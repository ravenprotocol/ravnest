import torch
import numpy as np
import time
import datasets
from .node import Node
from .utils import no_schedule
from .strings import *
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

class BaseTrainer():
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
    :param save: Whether to save the submodel after training, defaults to False.
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

    def __init__(self, node:Node = None, lr_scheduler=None, lr_scheduler_params={}, 
                 train_loader=None, val_loader=None, val_freq=1, save=False, epochs=1, 
                 batch_size=64, step_size=1, update_frequency = 1, loss_fn = None, accuracy_fn=None):
        self.node = node
        # if self.node.node_type == NodeTypes.STEM or self.node.node_type == NodeTypes.LEAF:
        #     return
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_freq = val_freq
        self.save = save
        self.epochs = epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.n_forwards = 0
        self.update_frequency = update_frequency
        self.lr_scheduler=None
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.node.optimizer, **lr_scheduler_params)

    def train(self):
        """Train the model using the specified training and validation data loaders.

        Iterates over the training data for the specified number of epochs,
        performing forward computations, updating parameters, and optionally
        evaluating on validation data.
        """
        # self.prelim_checks()
        t1 = time.time()
        for epoch in range(self.epochs):
            self.node.model.train()
            for X_train, y_train in self.train_loader:
                outputs = self.node.forward(X_train)
                loss = self.node.dist_func(self.loss_fn, args=(outputs, y_train))
                if self.node.node_type == NodeTypes.LEAF:
                    print('Loss: ', loss)
                self.node.backward(loss)

                if self.node.n_backwards % self.update_frequency == 0:
                    self.node.optimizer_step()
                    self.node.model.zero_grad()
                    self.node.optimizer.zero_grad()
            
            if self.val_loader is not None: 
                self.node.model.eval()
                acc = 0
                for X_test, y_test in self.val_loader:
                    output = self.node.no_grad_forward(X_test)
                    accuracy = self.node.dist_func(self.accuracy_fn, args=(output, y_test))
                    if self.node.node_type == NodeTypes.LEAF:
                        acc += accuracy.numpy()
                if self.node.node_type == NodeTypes.LEAF:
                    print('Accuracy: ', acc/len(self.val_loader))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            print('Epoch: ', epoch)

        self.node.model.train()
        self.await_backwards()
        
        self.node.comm_session.parallel_ring_reduce()
        print('Training Done!: ', time.time() - t1, ' seconds')

        if self.save:
            self.node.trigger_save_submodel()
    
    def await_backwards(self):
        while self.node.n_backwards < self.node.n_forwards:
            if self.node.node_type != NodeTypes.LEAF:
                self.node.backward()
                if self.node.n_backwards % self.update_frequency == 0:
                    self.node.optimizer_step()
                    self.node.model.zero_grad()
                    self.node.optimizer.zero_grad()   

    def pred(self, data):
        """Perform prediction on sample test data using the trained model.

        :param data: Sample data for prediction
        :type data: torch.Tensor
        :return: Prediction result
        :rtype: torch.Tensor
        """
        pred = self.node.no_grad_forward(data)
        return pred
    
    def evaluate(self):
        """Evaluate the trained model on the validation data.

        Performs inference on validation data and computes validation accuracy
        using the trained model.
        """

        if self.val_loader is not None: 
            self.node.model.eval()
            # if self.n_forwards % self.val_freq == 0:
            acc = 0
            for X_test, y_test in self.val_loader:
                # self.node.no_grad_forward_compute(tensors=X_test, output_type='val_accuracy')
                output = self.node.no_grad_forward(X_test)
                accuracy = self.criterion(self.accuracy_fn, args=(output, y_test))
                if accuracy is not None:
                    acc += accuracy.numpy()
            print('Accuracy: ', acc/len(self.val_loader))


class BaseTrainerFullAsync():
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
    :param save: Whether to save the submodel after training, defaults to False.
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

    def __init__(self, node:Node = None, lr_scheduler=None, lr_scheduler_params={}, 
                 train_loader=None, val_loader=None, val_freq=1, save=False, epochs=1, 
                 batch_size=64, step_size=1, update_frequency = 1, loss_fn = None, accuracy_fn=None):
        self.node = node
        # if self.node.node_type == NodeTypes.STEM or self.node.node_type == NodeTypes.LEAF:
        #     return
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_freq = val_freq
        self.save = save
        self.epochs = epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.n_forwards = 0
        self.update_frequency = update_frequency
        self.lr_scheduler=None
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.node.optimizer, **lr_scheduler_params)

    @no_schedule()
    def train_step(self, x, y):
        outputs = self.node.forward(x)
        loss = self.node.dist_func(self.loss_fn, args=(outputs, y))
        # if self.node.node_type == NodeTypes.LEAF:
        #     print('Loss: ', loss)
        self.node.backward(loss)

        if self.node.n_backwards % self.update_frequency == 0:
            self.node.optimizer_step()
            self.node.model.zero_grad()
            self.node.optimizer.zero_grad()

    def train(self):
        """Train the model using the specified training and validation data loaders.

        Iterates over the training data for the specified number of epochs,
        performing forward computations, updating parameters, and optionally
        evaluating on validation data.
        """
        # self.prelim_checks()
        t1 = time.time()
        for epoch in range(self.epochs):
            self.node.model.train()
            t2 = time.time()
            for X_train, y_train in self.train_loader:
                self.train_step(X_train, y_train)
            
            self.await_backwards()

            # if self.val_loader is not None: 
            #     # self.await_backwards()
            #     self.node.model.eval()
            #     acc = 0
            #     for X_test, y_test in self.val_loader:
            #         # self.await_one_backward()
            #         # self.node.model.eval()
            #         output = self.node.no_grad_forward(X_test)
            #         accuracy = self.node.dist_func(self.accuracy_fn, args=(output, y_test))
            #         if self.node.node_type == NodeTypes.LEAF:
            #             acc += accuracy.numpy()
            #     if self.node.node_type == NodeTypes.LEAF:
            #         print('Accuracy: ', acc/len(self.val_loader))
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            t_epoch = time.time() - t2
            print('Epoch: ', epoch, ' time taken: ', t_epoch)
        self.node.model.train()
        self.await_backwards()
        print('Training Done!: ', time.time() - t1, ' seconds')
        
        self.node.comm_session.parallel_ring_reduce()
        # print('Training Done!: ', time.time() - t1, ' seconds')

        if self.save:
            self.node.trigger_save_submodel()
    
    def await_backwards(self):
        while self.node.n_backwards < self.node.n_forwards:
            self.await_one_backward()

    def await_one_backward(self):
        if self.node.node_type != NodeTypes.LEAF:
            self.node.backward()
            if self.node.n_backwards % self.update_frequency == 0:
                self.node.optimizer_step()
                self.node.model.zero_grad()
                self.node.optimizer.zero_grad()  
    
    def pred(self, data):
        """Perform prediction on sample test data using the trained model.

        :param data: Sample data for prediction
        :type data: torch.Tensor
        :return: Prediction result
        :rtype: torch.Tensor
        """
        pred = self.node.no_grad_forward(data)
        return pred
    
    def evaluate(self):
        """Evaluate the trained model on the validation data.

        Performs inference on validation data and computes validation accuracy
        using the trained model.
        """

        if self.val_loader is not None: 
            self.node.model.eval()
            # if self.n_forwards % self.val_freq == 0:
            acc = 0
            for X_test, y_test in self.val_loader:
                # self.node.no_grad_forward_compute(tensors=X_test, output_type='val_accuracy')
                output = self.node.no_grad_forward(X_test)
                accuracy = self.criterion(self.accuracy_fn, args=(output, y_test))
                if accuracy is not None:
                    acc += accuracy.numpy()
            print('Accuracy: ', acc/len(self.val_loader))    