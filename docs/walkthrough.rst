Walkthrough: Decentralized CNN Training 
=======================================

This section explains how you can fire up ravnest to train a simple **CNN model** on **MNIST data** across **3 nodes** that will be hosted locally on your device.

Before proceeding , please make sure ravnest is installed in your python environment. 

Start off by creating a blank project directory.

Configuring the Provider Nodes
------------------------------

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

The above code defines 3 nodes, each having 8 GB RAM each and a network bandwidth of 10 Mbps. Since we will be spawning these 3 compute nodes locally, we set different ports for each node's IP address. This is our pool of Provider Nodes. 

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
    :emphasize-lines: 1,7,13

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


Provider Script
---------------

After completing the steps defined in :ref:`this<local-reference-label>` section, you will find all metadatas pertaining to each individual node in the ``node_data/nodes`` folder. In this case, you will find 3 files (``node_0.json``, ``node_1.json`` and ``node_2.json``).

Next up, you need to create the consolidated Provider script which incorporates a data preprocessing method, Node instance, and Trainer instance with the appropriate parameters:

.. code-block:: python
    :linenos:
    :emphasize-lines: 46

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

    def preprocess_dataset():
        data = datasets.load_digits()
        X = data.data
        y = data.target

        # Convert to one-hot encoding
        y = to_categorical(y.astype("int"))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

        # Reshape X to (n_samples, channels, height, width)
        X_train = X_train.reshape((-1, 1, 8, 8))
        X_test = X_test.reshape((-1, 1, 8, 8))

        generator = torch.Generator()
        generator.manual_seed(42)

        train_loader = DataLoader(list(zip(X_train,torch.tensor(y_train, dtype=torch.float32))), generator=generator, shuffle=True, batch_size=64)
        val_loader = DataLoader(list(zip(X_test,torch.tensor(y_test, dtype=torch.float32))), shuffle=False, batch_size=64)

        return train_loader, val_loader

    def loss_fn(preds, targets):
        return torch.nn.functional.mse_loss(preds, targets[1])

    if __name__ == '__main__':

        train_loader, val_loader = preprocess_dataset()

        node = Node(name = 'node_0', 
                    optimizer = torch.optim.Adam,
                    device=torch.device('cpu'),
                    criterion = loss_fn,
                    labels = train_loader, 
                    test_labels=val_loader
                    )

        trainer = Trainer(node=node,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        val_freq=64,
                        epochs=100,
                        batch_size=64,
                        inputs_dtype=torch.float32,
                        save=True)

        trainer.train()

        trainer.evaluate()

Create 3 files named ``provider_0.py``, ``provider_1.py`` and ``provider_2.py`` in your project directory. Copy and paste the above code in all 3 files. 

Now simply change the ``name`` passed to ``Node()`` (highlighted line) to ``'node_0'``, ``'node_1'`` and ```node_2'`` in ``provider_0.py``, ``provider_1.py`` and ``provider_2.py`` respectively. For your convenience, this line has been highlighted in the above code snippet.


Project Directory Structure
---------------------------

If you've been diligently following along, behold the splendid sight that is your project directory now:

.. code-block:: bash

    .
    ├── cluster_formation.py
    ├── models.py
    ├── node_data
    │   ├── cluster_0
    │   │   ├── 0.0.0.0:8080
    │   │   │   ├── model_inputs.pkl
    │   │   │   ├── submod.pt
    │   │   │   ├── submod_0_input.pkl
    │   │   │   └── submod_0_output.pkl
    │   │   ├── 0.0.0.0:8081
    │   │   │   ├── submod.pt
    │   │   │   ├── submod_1_input.pkl
    │   │   │   └── submod_1_output.pkl
    │   │   └── 0.0.0.0:8082
    │   │       ├── submod.pt
    │   │       ├── submod_2_input.pkl
    │   │       └── submod_2_output.pkl
    │   ├── node_configs.json
    │   └── nodes
    │       ├── node_0.json
    │       ├── node_1.json
    │       └── node_2.json
    ├── provider_0.py
    ├── provider_1.py
    └── provider_2.py

    6 directories, 19 files

If everything seems to be in place, you are ready to start off your Decentralized CNN Training Session on your Local System!

Executing Providers
-------------------

Simply open 3 terminals with your python virtual environment enabled and run the following commands in them:

.. code-block:: bash
    
    python provider_2.py

.. code-block:: bash
    
    python provider_1.py

.. code-block:: bash
    
    python provider_0.py

Monitoring Training Metrics
---------------------------

As training progresses, you can view the training losses in a ``losses.txt`` file that automatically gets created in your project directory. Additionally, you may also find a file named ``val_accuracies.txt`` that periodically logs the validation accuracy.

Retrieving Trained Final Model
------------------------------

Setting the ``save`` parameter of ``Trainer()`` instance to ``True`` in the Provider's script enables saving of corresponding submodels post-training. You can now combine these submodels across any cluster and have a complete cohesive state_dict that contains the trained weights and parameters. 

Run the following code to save this consolidated state_dict at ``trained/trained_state_dict.pt``:

.. code-block:: python
 
    import ravnest

    ravnest.model_fusion(cluster_id=0)

Adjust the ``cluster_id`` parameter to retrieve and save the trained state_dict for the respective cluster. 

With the consolidated state_dict from your decentralized training session, you can load it into your main PyTorch model using ``model.load_state_dict('trained/trained_state_dict.pt')``. You can run inference, deploy this trained model, or use it for any other purpose!
