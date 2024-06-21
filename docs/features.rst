Features
========

GPU Support
-----------

Ravnest significantly benefits from GPU acceleration, enhancing the performance and efficiency of model training across distributed environments. The platform seamlessly integrates GPU support for Provider nodes, leveraging their parallel processing capabilities to handle complex computations faster than traditional CPU-based machines. 

If a Provider has an NVIDIA GPU on their machine, they can enable GPU-acceleration by setting the ``device`` parameter of ``Node()`` object to ``torch.device('cuda')``. 


Custom Trainers
---------------

By subclassing the existing ``Trainer`` class from Ravnest, you can incorporate specialized logic for handling model training, validation, and metrics specific to your project's needs. This approach not only streamlines the process of integrating new models but also provides flexibility in adapting to diverse training scenarios and evolving requirements.

.. code-block:: python
    :linenos:    

    import ravnest

    class Custom_Trainer(ravnest.Trainer):
        def __init__(self, node=None, train_loader=None, epochs=1):
            super().__init__(node=node, train_loader=train_loader, epochs=epochs)

        # Overwrite the train() method as per your requirements:
        def train(self):
            self.prelim_checks()    # Mandatory function call at start of train() method
            '''
            Training Loop goes here.
            Use self.node.forward_compute() to perform forward pass. Pass an incrementing data_id to this function on every call.
            Use self.node.wait_for_backwards() at end of every epoch to uphold order of respective backward passes.
            '''
            ...


As an example, here's a custom Trainer class for pre-training BERT LLM that expects multiple tensors as inputs:

.. code-block:: python
    :linenos:
    :emphasize-lines: 3,8,10,12-15,16 

    import ravnest

    class BERT_Trainer(ravnest.Trainer):
        def __init__(self, node=None, train_loader=None, epochs=1):
            super().__init__(node=node, train_loader=train_loader, epochs=epochs)

        def train(self):
            self.prelim_checks()    # Mandatory function call
            for epoch in range(self.epochs):
                data_id = 0
                for batch in self.train_loader:
                    self.node.forward_compute(data_id=data_id, 
                                            tensors=batch['input_ids'], 
                                            l_token_type_ids_=batch['token_type_ids'], 
                                            l_attention_mask_=batch['attention_mask'])  
                    data_id += 1    # Increments at every batch

                self.node.wait_for_backwards()   # To be called at end of every epoch
                    
            print('BERT Training Done!')


Data Compression
----------------
Ravnest uses advanced data compression techniques to optimize distributed training. These techniques significantly decrease data transmission between provider nodes, reducing network overhead and improving training efficiency. By compressing model parameters and gradients, Ravnest enables faster communication and synchronization among nodes, leading to shorter training times. To activate this capability, providers can set ``compression=True`` in the ``Node()`` object.

This feature is particularly advantageous in bandwidth-constrained environments, maximizing resource utilization. Moreover, Ravnest's compression mechanism maintains the integrity and accuracy of the training process, safeguarding model performance.