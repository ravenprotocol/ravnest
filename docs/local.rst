Locally Training a Deep Learning Model
======================================

This section explains how you can set up a couple of scripts then fire up ravnest to train a simple **CNN model** on **MNIST data** across **3 nodes** that will be hosted locally on your device.

Start off by creating a blank project directory.

Configuring the Compute Nodes
-----------------------------

Ravnest requires a json file that defines the available RAM and Network Bandwidth for each participating node. 

In your project directory, create a json file at ``node_data/node_configs.json``. Add the following code to this json file:

.. code-block:: python
    :linenos:

    {
        "0":{
            "IP":"0.0.0.0:8080",
            "benchmarks":{
                "ram":128,
                "bandwidth":20
            }
        },
        "1":{
            "IP":"0.0.0.0:8081",
            "benchmarks":{
                "ram":128,
                "bandwidth":20
            }
        },
        "2":{
            "IP":"0.0.0.0:8082",
            "benchmarks":{
                "ram":64,
                "bandwidth":40
            }
        }
    }

Since we will be spawning these 3 compute nodes locally, we set different ports for each node's IP address. This is our pool of Nodes. 

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

.. _my-reference-label:

Forming Clusters from the Pool of Compute Nodes
-----------------------------------------------

Next, create a ``cluster_formation.py`` file with the following lines of code:

.. code-block:: python
    :linenos:
    
    from ravnest.operations.utils import clusterize
    from models import CNN_Net

    model = CNN_Net()

    clusterize(model=model)

Note that we have simply imported our CNN model from the correct file and passed it to the ``clusterize()`` method.

You will observe that running the above code ( with the command ``python cluster_formation.py``) spawns a few subfolders housing some metadata inside the ``node_data`` folder. 

Under the hood, Ravnest uses it's magic sauce Genetic Algorithm to optimally form clusters of compute nodes such that the nodes with similar capabilities get grouped together. Now depending on the complexity of your deep learning model and the total number of nodes you want to train on, Ravnest may form multiple clusters. With the values provided in this tutorial, you will see that one cluster containing 3 nodes has been formed. Feel free to play around with different models and number of nodes in the ``node_data/node_configs.json`` file to see it in action. 

At this stage, you are ready to start off your Decentralized Training Session on your Local System!

