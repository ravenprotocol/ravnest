import os
import torch
import json
import shutil
import random
import numpy as np
from ravnest.operations.utils import *

from models import CNN_Net

# from models import inception_v3

# from examples.sorter.mingpt.model_without_padding_mask import GPT
# from examples.sorter.dataset import SortDataset

# from torchvision.models import resnet50


random.seed(42)
np.random.seed(42)

def delete_all_folders(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

path = 'node_data/'
delete_all_folders(path)

# For CNN Model    
model = CNN_Net()

## For ResNET 50 Model
# model = resnet50(num_classes=200)

# # For Inception V3 Model
# model = inception_v3()

## For Sorter Model
# train_dataset = SortDataset('train')
# test_dataset = SortDataset('test')
# model_config = GPT.get_default_config()
# model_config.model_type = 'gpt-nano'
# model_config.vocab_size = train_dataset.get_vocab_size()
# model_config.block_size = train_dataset.get_block_size()
# model = GPT(model_config)

node_pool = spawn_node_pool(mode='load_from_configs')

cluster_pool = cluster_formation(full_model_size=len(model.state_dict()), node_pool=node_pool)

for node in node_pool:
    node_path = 'node_data/cluster_{}/{}'.format(node.cluster_id, node.address)
    if not os.path.exists(node_path):
        os.makedirs(node_path)
    if not os.path.exists('node_data/nodes'):
        os.makedirs('node_data/nodes')

for cluster in cluster_pool:
    model_input_node = cluster.nodes[list(cluster.nodes.keys())[0]].address
    cluster_node_ip_addresses = []
    for node_id, metadata in cluster.nodes.items():
        cluster_node_ip_addresses.append(metadata.address)
    
    for i in range(len(cluster_node_ip_addresses)):
        for node in node_pool:
            if node.address == cluster_node_ip_addresses[i]:
                current_node = node
                break
        if i < len(cluster_node_ip_addresses) - 1:
            current_node.forward_target_host = cluster_node_ip_addresses[i+1].split(':')[0]
            current_node.forward_target_port = cluster_node_ip_addresses[i+1].split(':')[1]
        if i > 0 and i < len(cluster_node_ip_addresses):
            current_node.backward_target_host = cluster_node_ip_addresses[i-1].split(':')[0]
            current_node.backward_target_port = cluster_node_ip_addresses[i-1].split(':')[1]

    split_model_equal(model=model,
                      num_splits=len(cluster.nodes), 
                      cluster_path='node_data/cluster_{}'.format(cluster.cid), 
                      node_paths=cluster_node_ip_addresses,
                      model_input_node = model_input_node)
    
for node in node_pool:
    node.set_submodel()
    # print('\n Node id: ', node.node_id, ' params: ', node.submodel.state_dict().keys())

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

max_ring_size = {} 
for cluster in cluster_pool:
    if cluster.cid != max_c.cid:
        for nid, node in cluster.nodes.items():
            current_ring_id = None
            node.set_trainable_parameter_keys()
            for k in node.trainable_param_keys:
                if current_ring_id != max_c.all_param_to_ring[k]:
                    node.ring_ids[max_c.all_param_to_ring[k]] = k
                    current_ring_id = max_c.all_param_to_ring[k]

    for _,node in cluster.nodes.items():
        print(node.ring_ids)
        for key in node.ring_ids:
            max_ring_size[key] = max_ring_size.get(key, 0) + 1
    
max_ring_size_value = max(max_ring_size.values())

for cl in range(len(cluster_pool)):
    cluster = cluster_pool[cl]
    if cl == len(cluster_pool) - 1:
        next_cluster = cluster_pool[0]
    else:
        next_cluster = cluster_pool[cl + 1]

    for nid, node in cluster.nodes.items():
        current_address = None
        for k in node.trainable_param_keys:
            for n_nid, n_node in next_cluster.nodes.items():
                if k in n_node.trainable_param_keys:
                    if current_address != n_node.address:
                        node.address_to_param[n_node.address] = k
                        current_address = n_node.address
    
print(node_pool)
print(cluster_pool)

for node in node_pool:
    node_meta = {}
    node_meta['node_id'] = node.node_id
    node_meta['local_host'] = node.address.split(':')[0]
    node_meta['local_port'] = int(node.address.split(':')[1])
    node_meta['template_path'] = 'node_data/cluster_{}/{}/'.format(node.cluster_id, node.address)
    node_meta['rank'] = node.cluster_id
    node_meta['ring_size'] = max_ring_size_value
    node_meta['param_addresses'] = node.address_to_param,
    node_meta['ring_ids'] = {int(key): value for key, value in node.ring_ids.items()}
    node_meta['forward_target_host'] = node.forward_target_host
    node_meta['forward_target_port'] = int(node.forward_target_port) if node.forward_target_port is not None else None
    node_meta['backward_target_host'] = node.backward_target_host
    node_meta['backward_target_port'] = int(node.backward_target_port) if node.backward_target_port is not None else None

    with open('node_data/nodes/node_{}.json'.format(node.node_id), 'w') as fp:
        json.dump(node_meta, fp)
