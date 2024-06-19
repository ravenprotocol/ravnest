Walkthrough: Local Decentralized Training 
=========================================

This section explains how you can fire up ravnest to train a simple **CNN model** on **MNIST data** across **3 nodes** that will be hosted locally on your device.

Before proceeding , please make sure ravnest is installed in your python environment. 

Start off by creating a blank project directory.

Configuring the Compute Nodes
-----------------------------

Ravnest requires a json file that defines the available RAM (in GBs) and Network Bandwidth (in Mbps) for each participating node. 

In your project directory, create a json file at ``node_data/node_configs.json``. Add the following code to this json file:

.. code-block:: json
    :linenos:

    {
        "0":{
            "IP":"0.0.0.0:8080",
            "benchmarks":{
                "ram":8,
                "bandwidth":10
            }
        },
        "1":{
            "IP":"0.0.0.0:8081",
            "benchmarks":{
                "ram":8,
                "bandwidth":10
            }
        },
        "2":{
            "IP":"0.0.0.0:8082",
            "benchmarks":{
                "ram":8,
                "bandwidth":10
            }
        }
    }

The above code defines 3 nodes, each having 8 GB RAM each and a network bandwidth of 10 Mbps. Since we will be spawning these 3 compute nodes locally, we set different ports for each node's IP address. This is our pool of Nodes. 

Defining the Deep Learning Model
--------------------------------

Next, let's create a ``models.py`` file and define our CNN Pytorch Model in it:

.. code-block:: python
    :linenos:
    
    import torch.nn as nn

    class CNN_Net(nn.Module):    
        def __init__(self):
            super(CNN_Net, self).__init__()
            self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding='same')
            self.act_1 = nn.ReLU()
            self.maxpool2d_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            self.drp_1 = nn.Dropout(0.25)
            self.bn_1 = nn.BatchNorm2d(16)
            self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
            self.act_2 = nn.ReLU()
            self.maxpool2d_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            self.drp_2 = nn.Dropout(0.25)
            self.bn_2 = nn.BatchNorm2d(32)
            self.flatten = nn.Flatten()
            self.dense_1 = nn.Linear(in_features=32,out_features=256)
            self.act_3 = nn.ReLU()
            self.drp_3 = nn.Dropout(0.4)
            self.bn_3 = nn.BatchNorm1d(256)
            self.dense_2 = nn.Linear(in_features=256, out_features=10)
            self.act_4 = nn.Softmax(dim=-1)

        def forward(self, x):
            out = self.conv2d_1(x)
            out = self.act_1(out)
            out = self.maxpool2d_1(out)
            out = self.drp_1(out)
            out = self.bn_1(out)
            out = self.maxpool2d_2(out)
            out = self.conv2d_2(out)
            out = self.act_2(out)
            out = self.maxpool2d_3(out)
            out = self.drp_2(out)
            out = self.bn_2(out)
            out = self.flatten(out)
            out = self.dense_1(out)
            out = self.act_3(out)
            out = self.drp_3(out)
            out = self.bn_3(out)
            out = self.dense_2(out)
            out = self.act_4(out)
            return out

.. _local-reference-label:

Forming Clusters from the Pool of Compute Nodes
-----------------------------------------------

Next, create a ``cluster_formation.py`` file with the following lines of code:

.. code-block:: python
    :linenos:
    
    import torch
    from ravnest import clusterize, set_seed
    from models import CNN_Net

    set_seed(42)

    model = CNN_Net()
    example_args = torch.rand((64,1,8,8))
    clusterize(model=model, example_args=(example_args,))

We have simply imported our CNN model from the ``models.py`` file and passed it to the ``clusterize()`` method, along with a set of ``example_args`` that enables Ravnest to calculate an estimate for the maximum memory that will ideally be required be train this model. Note that ``example_args`` is a simple random PyTorch Tensor having the exact shape and dtype that the ``CNN_Net`` model expects as input.

You will observe that running the above code (with the command ``python cluster_formation.py``) spawns a few subfolders housing some metadata inside the ``node_data`` folder. 

Under the hood, Ravnest uses it's awesomesauce Genetic Algorithm to optimally form clusters of compute nodes such that the nodes with similar capabilities get grouped together. Now depending on the complexity of your deep learning model and the total number of nodes you want to train on, Ravnest may form multiple clusters. With the values provided in this tutorial, you will see that one cluster containing 3 nodes has been formed. Feel free to play around with different models and number of nodes in the ``node_data/node_configs.json`` file to see it in action. 

The following logs that are generated upon executing the ``clusterize()`` method indicate that Node(0) is Root, Node(1) is Stem and Node(2) is Leaf:

.. code-block:: text
    :linenos:

    Node(0, Cluster(0)) 
    self.IP(0.0.0.0:8080) 
    Ring IDs({0: 'L__self___conv2d_1.weight'}) 
    Address2Param({'0.0.0.0:8080': 'L__self___conv2d_1.weight'})


    Node(1, Cluster(0)) 
    self.IP(0.0.0.0:8081) 
    Ring IDs({1: 'L__self___dense_1.weight'}) 
    Address2Param({'0.0.0.0:8081': 'L__self___dense_1.weight'})


    Node(2, Cluster(0)) 
    self.IP(0.0.0.0:8082) 
    Ring IDs({2: 'L__self___bn_3.weight'}) 
    Address2Param({'0.0.0.0:8082': 'L__self___bn_3.weight'})


Next up, you need to create the Provider scripts for Root, Stem and Leaf nodes.

Provider Scripts
----------------

Even though you can easily infer the roles of each Provider nodes based on the logs of ``clusterize()`` method, an alternate technique has been mentioned for each in the aforementioned subsections:

Root Node
~~~~~~~~~

After completing the steps defined in :ref:`this<local-reference-label>` section, you will find all metadatas pertaining to each individual node in the ``node_data/nodes`` folder. In this case, you will find 3 files (``node_0.json``, ``node_1.json`` and ``node_2.json``). 

You can identify the root node, by finding the json file where the ``backward_target_host`` and ``backward_target_port`` keys are set to ``null`` value. The name of that json file (in this case ``node_0``) is the ``name`` that you need to provide to ``Node()`` instance in the code below (to be saved as ``client_0.py``): 


.. code-block:: python
   :linenos:

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

    train_loader = DataLoader(list(zip(X,y)), shuffle=False, batch_size=64)
    val_loader = DataLoader(list(zip(X_test,y_test)), shuffle=False, batch_size=64)

    if __name__ == '__main__':

        node = Node(name = 'node_0', 
                    optimizer = torch.optim.Adam,
                    device=torch.device('cpu')
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


The Root Node needs to have the ``optimizer`` and ``train_loader`` defined and passed to the ``Node()`` and ``Trainer()`` class instances respectively. This Root Node is essentially the entry gateway for the training/validation data to flow into the cluster.  


Stem Node
~~~~~~~~~

You can identify the stem nodes by looking at the json files in ``node_data/nodes`` folder that has all 4 keys (``forward_target_host`` , ``forward_target_port``, ``backward_target_host`` and ``backward_target_port``) set to some non-null values (in our case ``node_1``). Use the following code to start your Stem Node (to be saved as ``client_1.py``):

.. code-block:: python
   :linenos:

    import torch
    import time
    from ravnest import Node, set_seed

    set_seed(42)

    if __name__ == '__main__':
            
        node = Node(name = 'node_1',
                    optimizer = torch.optim.Adam, 
                    device=torch.device('cpu')
                    )

        while True:
            time.sleep(0)


The optimizer needs to be same as the one used for the Root Node and passed to the instance of the ``Node`` class in the above code.

Leaf Node
~~~~~~~~~

You can easily identify the Leaf Node by looking at the json files in ``node_data/nodes`` folder. The file that has ``forward_target_host`` and ``forward_target_port`` set to ``null`` is the Leaf Node (in our case ``node_2``). Code for Leaf Node (to be saved as ``client_2.py``):

.. code-block:: python
   :linenos:

    import torch
    import numpy as np
    import time
    from sklearn import datasets
    from torch.utils.data import DataLoader
    from ravnest import Node, set_seed
    from sklearn.model_selection import train_test_split

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
        
        node = Node(name = 'node_2',
                    optimizer = torch.optim.Adam,
                    criterion = torch.nn.functional.mse_loss, 
                    labels = train_loader, 
                    test_labels=val_loader,
                    device=torch.device('cpu')
                    )
        
        while True:
            time.sleep(0)


The above code for the Leaf Node includes preprocessing steps for the training labels and the validation labels. Additionally, it also requires a ``criterion`` to be defined and passed to the instance of the ``Node`` class. 

Project Directory Structure
---------------------------

If you've been diligently following along, behold the splendid sight that is your project directory now:

.. code-block:: bash

    .
    ├── client_0.py
    ├── client_1.py
    ├── client_2.py
    ├── cluster_formation.py
    ├── models.py
    └── node_data
        ├── cluster_0
        │   ├── 0.0.0.0:8080
        │   │   ├── model_inputs.pkl
        │   │   ├── submod.pt
        │   │   ├── submod_0_input.pkl
        │   │   └── submod_0_output.pkl
        │   ├── 0.0.0.0:8081
        │   │   ├── submod.pt
        │   │   ├── submod_1_input.pkl
        │   │   └── submod_1_output.pkl
        │   └── 0.0.0.0:8082
        │       ├── submod.pt
        │       ├── submod_2_input.pkl
        │       └── submod_2_output.pkl
        ├── node_configs.json
        └── nodes
            ├── node_0.json
            ├── node_1.json
            └── node_2.json

If everything seems to be in place, you are ready to start off your Decentralized CNN Training Session on your Local System!

Executing Providers
-------------------

Simply open 3 terminals with your python virtual environment enabled and run the following commands in them:

.. code-block:: bash
    
    python client_2.py

.. code-block:: bash
    
    python client_1.py

.. code-block:: bash
    
    python client_0.py

Monitoring Training Metrics
---------------------------

As training progresses, you can view the training losses in a ``losses.txt`` file that automatically gets created in your project directory. Additionally, you may also find a file named ``val_accuracies.txt`` that periodically logs the validation accuracy.