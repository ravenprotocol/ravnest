Decentralized Training on Ravnest
=================================

Ravnest integrates seamlessly with PyTorch, providing a user-friendly interface that simplifies the setup and management of distributed training environments. It allows users to easily scale their training workloads across multiple devices, reducing training times and improving model performance without extensive configuration or specialized knowledge in distributed systems. By democratizing access to powerful training capabilities, Ravnest makes advanced machine learning more accessible and efficient for developers and researchers.

Getting the Main Model Ready
----------------------------

First, define your PyTorch model in the usual way. Ensure it is encapsulated within a class that inherits from ``torch.nn.Module`` and includes a defined ``forward()`` method.

.. code-block:: python
    :linenos:
    
    import torch
    import torch.nn as nn

    class DL_Model(nn.Module):
        def __init__(self):
            super(DL_Model, self).__init__()
            # Model layers go here
            ...

        def forward(self, x):
            # Forward pass logic goes here
            ...

    model = DL_Model()

Or, if you're feeling adventurous, load popular models straight from PyTorch-based libraries:

.. code-block:: python
    :linenos:
    
    from torchvision.models import resnet50
    model = resnet50()

For a scripted affair, load a ``.pt`` file using ``torch.jit.load()``:

.. code-block:: python
    :linenos:
    
    import torch
    model = torch.jit.load("model_scripted.pt")

As long as you have a PyTorch model instance, you're all set.

Defining the Provider Nodes
---------------------------

To set up the compute provider nodes, craft a JSON file containing the system specification metadata of each node. In your project directory, create a json file at ``node_data/node_configs.json``.

This file should include information such as the node's publicly accessible IP address, available compute memory (can be RAM or VRAM), and network bandwidth details. Below is an example structure of the JSON file:

.. code-block:: json
    :linenos:

    {
        "0":{
            "IP":"<host>:<port>",
            "benchmarks":{
                "ram":8,
                "bandwidth":20
            }
        },
        "1":{
            "IP":"<host>:<port>",
            "benchmarks":{
                "ram":16,
                "bandwidth":10
            }
        },
        "2":{
            "IP":"<host>:<port>",
            "benchmarks":{
                "ram":8,
                "bandwidth":10
            }
        }
        // Add as many nodes as you need here
    }

The ``ram`` values for each node, which can signify either CPU RAM or GPU VRAM, are mentioned in GBs while the ``bandwidth`` is in Mbps.

.. note::
    In future releases, we intend to implement mechanisms to dynamically create and update this JSON file as and when new nodes join the training session. For now, we can work by manually defining the provider nodes in the above format.  

Model Fragmentation and Cluster Formation
-----------------------------------------

The next step is to first form clusters of provider nodes followed by fragmentation of the main PyTorch model into sub-models and assigning them to individual provider nodes. Ravnest handles model fragmentation and orchestrates the cluster formation simultaneously, ensuring an optimal distribution of model parameters and computational load across the available provider nodes.

To achieve this, Ravnest needs to have a good estimation of how much maximum memory usage the model will require. This information is crucial for ensuring optimal cluster formation. Therefore, we pass a dummy input along with the main PyTorch model into Ravnest's ``clusterize()`` method.

.. code-block:: python
    :linenos:

    import torch
    from ravnest import clusterize, set_seed

    set_seed(42)

    model = DL_Model()    # The main PyTorch model which was previously defined/loaded.
    example_args = torch.rand((64,3,28,28))    # Sample input that the main PyTorch model expects. 
    
    clusterize(model=model, example_args=(example_args,))

For reproducibility, we encourage you to use ``set_seed()`` method. Running the above code spawns a few subfolders housing some metadata inside the ``node_data`` folder. If you explore the metadata, you will be able to spot the resultant sub-models. 

Inferring Provider Roles
------------------------

The cluster assigned to each individual provider node will be visible in the logs of ``clusterize()`` method. For instance: 

.. code-block:: text
    :linenos:
    :emphasize-lines: 1,7,13,19,25,31
    
    Node(0, Cluster(1)) 
    self.IP(0.0.0.0:8080) 
    Ring IDs({0: 'L__self___conv2d_1.weight'}) 
    Address2Param({'0.0.0.0:8081': 'L__self___conv2d_1.weight'})


    Node(1, Cluster(0)) 
    self.IP(0.0.0.0:8081) 
    Ring IDs({0: 'L__self___conv2d_1.weight'}) 
    Address2Param({'0.0.0.0:8080': 'L__self___conv2d_1.weight'})


    Node(2, Cluster(0)) 
    self.IP(0.0.0.0:8082) 
    Ring IDs({1: 'L__self___dense_1.weight'}) 
    Address2Param({'0.0.0.0:8083': 'L__self___dense_1.weight'})


    Node(3, Cluster(1)) 
    self.IP(0.0.0.0:8083) 
    Ring IDs({1: 'L__self___dense_1.weight'}) 
    Address2Param({'0.0.0.0:8082': 'L__self___dense_1.weight'})


    Node(4, Cluster(1)) 
    self.IP(0.0.0.0:8084) 
    Ring IDs({2: 'L__self___bn_3.weight'}) 
    Address2Param({'0.0.0.0:8085': 'L__self___bn_3.weight'})


    Node(5, Cluster(0)) 
    self.IP(0.0.0.0:8085) 
    Ring IDs({2: 'L__self___bn_3.weight'}) 
    Address2Param({'0.0.0.0:8084': 'L__self___bn_3.weight'})

From the above log, by looking at the order of node assignment for each cluster, the following can be inferred:

.. code-block:: text
    :linenos:

    Cluster 0 : Node(1) -> Node(2) -> Node(5)
    Cluster 1 : Node(0) -> Node(3) -> Node(4)

This makes it easy to identify the roles of each Provider node:

.. code-block:: text
    :linenos:

    Node(0) -> Root
    Node(1) -> Root
    Node(2) -> Stem
    Node(3) -> Stem
    Node(4) -> Leaf
    Node(5) -> Leaf

Preparing the Provider Script
-----------------------------

Now that the main model has been divided into sub-models and provider nodes have been organized into clusters, we can prepare the unified code that each Provider needs to execute according to their position within their designated cluster. The responsibilities and characteristics of the different roles that Providers can take up within a cluster have been covered in detail :ref:`here<provider-reference-label>`. 

Here's the template for creating a unified Provider script:

.. code-block:: python
    :linenos:

    import torch
    from torch.utils.data import DataLoader
    from ravnest import Node, Trainer, set_seed

    set_seed(42)

    def preprocess_dataset():
        """
        Method to pre-process the dataset.
        Returns PyTorch DataLoader Objects for Training and Validation with torch.Generator() object passed if shuffle=True.
        """
        ...
        return train_loader, val_loader

    if __name__ == '__main__':

        train_loader, val_loader = preprocess_dataset()

        node = Node(name='node_<id>', ...)   # Pass appropriate parameters to define your Node, including optimizer, criterion, labels, test_labels etc.

        trainer = Trainer(node=node, ...)     # Pass appropriate parameters like epochs, train_loader, val_loader etc. Can also be a Custom Trainer class instance that extends Ravnest's Trainer.

        trainer.train()     # Commences Training
        trainer.evaluate()  # To check accuracy of model post-training.

.. note::
    Please ensure that the correct ``name`` is passed to the ``Node()`` instance as a string in the format : ``'node_0'``, ``'node_7'``. Ravnest automatically determines the Provider's role based on this ``name`` parameter, so accuracy is essential.

In decentralized training, it is crucial that the data order is synchronized across all nodes to maintain the integrity of the training process. Since the training loss is ultimately evaluated at the Leaf node (another type of node present at the end of the cluster), the data instances processed by the Root Provider must match those processed by the Leaf Provider. 

For the training to be accurate, the order of data instances in the ``DataLoader`` used by the Root Provider must be identical to the order in the ``DataLoader`` used by the Leaf Provider. This synchronization ensures that each data instance is paired with the correct true label during training, which is essential for the model to learn correctly. To ensure this, we utilize Ravnest's ``set_seed()`` method and pass the same seed value across all Provider scripts in a cluster. 

Incase you intend to employ data shuffling inside the DataLoader, we strongly encourage you to additionally define a ``torch.Generator()`` object and pass it on to your ``DataLoader`` instance. Doing so helps maintain the order of the data instances when ``shuffle=True``.

In the template provided above, ensure to include the ``labels`` and ``test_labels`` parameters (as ``DataLoader`` instances) when initializing ``Node()``, enabling accurate evaluation of training and validation losses with the correct labels. 

Refer to the :ref:`API Documentation<api-reference-label>` section for details on the other parameters that are required by ``Node`` and ``Trainer``.
