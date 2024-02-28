import torch
import numpy as np
import time

class Trainer():
    def __init__(self, node=None, train_loader=None, val_loader=None, val_freq=1, save=False, epochs=1, batch_size=64, step_size=1, inputs_dtype=None):
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
                    if self.n_forwards % self.val_freq == 0:
                        for X_test, y_test in self.val_loader:
                            self.node.no_grad_forward_compute(tensors=torch.tensor(X_test.numpy(), dtype=self.inputs_dtype), output_type='val_accuracy')

            print('Epoch: ', epoch)
            # print('n_forward: ', self.n_forwards, '  node.n_backward: ', self.node.n_backwards)
        
        while self.node.n_backwards < self.n_forwards:
            time.sleep(1)
        
        print('Training Done!: ', time.time() - t1, ' seconds')

        if self.save:
            self.node.trigger_save_submodel()
        
    def pred(self, input):
        if isinstance(input, np.ndarray):
            pred = self.node.no_grad_forward_compute(tensors=torch.tensor(input, dtype=torch.float32), output_type='accuracy')
        else:
            pred = self.node.no_grad_forward_compute(tensors=torch.tensor(input.numpy(), dtype=torch.float32), output_type='accuracy')
        return pred