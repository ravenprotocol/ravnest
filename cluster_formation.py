import os
import torch
import json
import shutil
import random
import numpy as np
from cluster_node_operations.misc import *
from cluster_node_operations.utils import *

random.seed(42)
np.random.seed(42)

def delete_all_folders(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

path = 'node_data/'
delete_all_folders(path)

class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
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
        # print(out.shape)

        out = self.maxpool2d_2(out)
        # print(out.shape)

        out = self.conv2d_2(out)
        out = self.act_2(out)
        out = self.maxpool2d_3(out)
        # print(out.shape)

        out = self.drp_2(out)
        out = self.bn_2(out)
        # print(out.shape)
        out = self.flatten(out)
        out = self.dense_1(out)
        out = self.act_3(out)
        out = self.drp_3(out)
        out = self.bn_3(out)
        out = self.dense_2(out)
        out = self.act_4(out)
        return out
    
model = Net()

node_pool = spawn_node_pool(mode='load_from_configs')

cluster_pool = cluster_formation(full_model_size=len(model.state_dict()), node_pool=node_pool)

for node in node_pool:
    node_path = 'node_data/cluster_{}/{}'.format(node.cluster_id, node.address)
    if not os.path.exists(node_path):
        os.makedirs(node_path)

for cluster in cluster_pool:
    cluster_node_ip_addresses = []
    for node_id, metadata in cluster.nodes.items():
        cluster_node_ip_addresses.append(metadata.address)
    split_model_equal(model=model,
                      num_splits=len(cluster.nodes), 
                      cluster_path='node_data/cluster_{}'.format(cluster.cid), 
                      node_paths=cluster_node_ip_addresses)
    
for node in node_pool:
    node.set_submodel()
    print('\n Node id: ', node.node_id, ' params: ', node.submodel.state_dict().keys())

max_c = None
max_l = 0
for cluster in cluster_pool:
    if cluster.size > max_l:
        max_l = cluster.size
        max_c = cluster

print('No. rings: ', max_l)
rid = 0
for nid, node in max_c.nodes.items():
    node.set_trainable_parameter_keys()
    node.ring_ids[rid] = node.trainable_param_keys[0]
    rid += 1

max_c.set_ringwise_params()
print('For max_c: ', max_c.cid)
for nid,node in max_c.nodes.items():
    print('node id: ', node.node_id, node.ring_ids)

print('maxc all param to ring')
print(max_c.all_param_to_ring)

for cluster in cluster_pool:
    if cluster.cid != max_c.cid:
        for nid, node in cluster.nodes.items():
            current_ring_id = None
            node.set_trainable_parameter_keys()
            for k in node.trainable_param_keys:
                if current_ring_id != max_c.all_param_to_ring[k]:
                    node.ring_ids[max_c.all_param_to_ring[k]] = k
                    current_ring_id = max_c.all_param_to_ring[k]

    print('\nRing ids cid: ', cluster.cid)
    print([node.ring_ids for _,node in cluster.nodes.items()])  

for cl in range(len(cluster_pool)):
    cluster = cluster_pool[cl]
    if cl == len(cluster_pool) - 1:
        next_cluster = cluster_pool[0]
    else:
        next_cluster = cluster_pool[cl + 1]

    for nid, node in cluster.nodes.items():
        current_address = None
        for k in node.trainable_param_keys:
            # print(k)
            for n_nid, n_node in next_cluster.nodes.items():
                if k in n_node.trainable_param_keys:
                    # print(current_address, n.address)
                    if current_address != n_node.address:
                        node.address_to_param[n_node.address] = k
                        current_address = n_node.address
    
    print('\nCluster addresses for cid: ', cluster.cid)
    print([node.address_to_param for _,node in cluster.nodes.items()])

print(node_pool)

print(cluster_pool)