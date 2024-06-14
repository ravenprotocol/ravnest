import torch
import numpy as np
import time

class Trainer():
    def __init__(self, node=None, train_loader=None, val_loader=None, save=False, epochs=1, batch_size=64, step_size=1, inputs_dtype=None):
        self.node = node
        self.train_loader = train_loader

        self.val_loader = val_loader
        self.save = save
        self.epochs = epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.n_forwards = 0
        self.inputs_dtype = inputs_dtype
        self.device = node.device

    def train(self):
        t1 = time.time()
        self.n_forwards = 0
        for epoch in range(self.epochs):
            data_id = 0
            for data in self.train_loader:
                # ip = data.to(self.device)
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                token_type_ids = data['token_type_ids']
                
                self.node.forward_compute(data_id=data_id, tensors=attention_mask, l_input_ids_=input_ids, l_token_type_ids_=token_type_ids)
                self.n_forwards += 1                
                data_id += (self.batch_size // self.step_size)

            self.node.wait_for_backwards()
            print('---------------- Epoch: ', epoch, ' ----------------')
            # print('n_forward: ', self.n_forwards, '  node.n_backward: ', self.node.n_backwards)
        
        print('Training Done!: ', time.time() - t1, ' seconds')

        if self.save:
            self.node.trigger_save_submodel()
