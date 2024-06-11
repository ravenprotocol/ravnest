Defining Compute Nodes 
======================

All participating nodes in Ravnest are required to take on one of the following roles during a training session:

- Root Node
- Middle Node
- Leaf Node

Within a cluster, there can be multiple middle nodes, however, only a single root node and a single leaf node must exist.

Root Node
---------

After completing the steps defined in :ref:`this<my-reference-label>` section, you will find all metadatas pertaining to each individual node in the ``node_data/nodes`` folder. In this case, you will find 3 files (``node_0.json``, ``node_1.json`` and ``node_2.json``). 

Now, to identify which node is a root node, simply find the json file where the ``backward_target_host`` and ``backward_target_port`` keys are set to ``null`` value. The name of that json file (in this case ``node_0``) is the ``node_name`` that you need to provide in the code below: 


.. code-block:: python
   :linenos:

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
    val_loader = DataLoader(list(zip(X_test,y_test)), shuffle=False, batch_size=64)

    if __name__ == '__main__':

        node_name = 'node_0'

        node_metadata = load_node_json_configs(node_name=node_name)
        model = torch.jit.load(node_metadata['template_path']+'submod.pt')
        optimizer=torch.optim.Adam
        
        node = Node(name = node_name, 
                    model = model, 
                    optimizer = optimizer,
                    device=torch.device('cpu'),
                    **node_metadata
                    )

        trainer = Trainer(node=node,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        val_freq=64,
                        epochs=100,
                        batch_size=64,
                        inputs_dtype=torch.float32)

        trainer.train()

The above Root Node code contains the data preprocessing steps as well. 

.. note::

    The Root Node needs to have the ``optimizer`` and ``train_loader`` defined. Both need to be passed to the ``Node`` class instance. This Root Node is essentially the entry gateway for the training/validation data to flow into the cluster.  


Middle Node
-----------

You can identify the middle nodes by looking at the json files in ``node_data/nodes`` folder that has all 4 keys (``forward_target_host`` , ``forward_target_port``, ``backward_target_host`` and ``backward_target_port``) set to some non-null values (in our case ``node_1``). Use the following code to start your Middle Node:

.. code-block:: python
   :linenos:

    import torch
    import time
    from ravnest.node import Node
    from ravnest.utils import load_node_json_configs

    if __name__ == '__main__':
        
        node_name = 'node_1'

        node_metadata = load_node_json_configs(node_name=node_name)
        model = torch.jit.load(node_metadata['template_path']+'submod.pt')
        optimizer=torch.optim.Adam
        
        node = Node(name = node_name, 
                    model = model, 
                    optimizer = optimizer, 
                    device=torch.device('cpu'),
                    **node_metadata
                    )

        while True:
            time.sleep(1)

.. note::

    The optimizer needs to be same as the one used for the Root Node and passed to the instance of the ``Node`` class in the above code.

Leaf Node
---------

You can easily identify the Leaf Node by looking at the json files in ``node_data/nodes`` folder. The file that has ``forward_target_host`` and ``forward_target_port`` set to ``null`` is the Leaf Node (in our case ``node_2``). Code for Leaf Node:

.. code-block:: python
   :linenos:

    import torch
    import numpy as np
    import time
    from sklearn import datasets
    from torch.utils.data import DataLoader
    from ravnest.node import Node
    from ravnest.utils import load_node_json_configs
    from sklearn.model_selection import train_test_split


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
        
        node_name = 'node_2'

        node_metadata = load_node_json_configs(node_name=node_name)
        model = torch.jit.load(node_metadata['template_path']+'submod.pt')
        optimizer=torch.optim.Adam
        criterion = torch.nn.functional.mse_loss

        node = Node(name = node_name, 
                    model = model, 
                    optimizer = optimizer,
                    criterion = criterion, 
                    labels = train_loader, 
                    test_labels=val_loader,
                    device=torch.device('cpu'),
                    **node_metadata
                    )
        
        while True:
            time.sleep(1)

.. note::

    The above code for the Leaf Node includes preprocessing steps for the training labels and the validation labels. Additionally, it also requires a ``criterion`` to be defined and passed to the instance of the ``Node`` class. 

