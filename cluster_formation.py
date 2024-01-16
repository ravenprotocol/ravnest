from cluster_node_operations.utils import *
import torch
from torch import nn
import os
import shutil


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

all_params = list(model.state_dict().keys())
# print(all_params)

full_model_size = len(model.state_dict().keys())

node_pool = spawn_node_pool(mode='load_from_configs')
cluster_pool = cluster_formation(full_model_size=full_model_size, node_pool=node_pool, state_dict=list(model.state_dict().keys()))

for node in node_pool:
    node_path = 'node_data/cluster_{}/{}'.format(node.cluster.cluster_id, node.ip_address)
    if not os.path.exists(node_path):
        os.makedirs(node_path)
    
for cluster in cluster_pool:
    cluster_node_ip_addresses = []
    for node_id, metadata in cluster.nodes.items():
        cluster_node_ip_addresses.append(metadata.ip_address)
    split_model_equal(model=model, 
                      num_splits=len(cluster.nodes), 
                      cluster_path='node_data/cluster_{}'.format(cluster.cluster_id), 
                      node_paths=cluster_node_ip_addresses)

formed_rings = form_rings(cluster_pool=cluster_pool)

for node in node_pool:
    node.set_submodel()
    node.set_ring_id_to_named_param_mapping()

assigned_connection_targets = assign_connection_targets(cluster_pool=cluster_pool)

print(node_pool)
print(cluster_pool)