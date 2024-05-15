import torch
import setuptools
import pickle
from pip._internal.operations.freeze import freeze

from torch.fx import Tracer
from pippy.IR import Pipe
from pippy import split_into_equal_size

from .pippy_utils import split_on_proportions
from .genetic import genetic_algorithm
from .cluster import Cluster
from .node import Node
import numpy as np
import random
import json
import os
import shutil
import copy

def spawn_node_pool(num_nodes=None, mode=None, ram_variants=None, bandwidth_variants=None):
    node_pool = []
    if mode == 'load_from_configs':
        file = open('node_data/node_configs.json')
        node_configs = json.load(file)
        num_nodes = len(node_configs.keys())
        for nid in range(num_nodes):
            # print(nid, node_configs[str(nid)]['IP'], node_configs[str(nid)]['benchmarks'])
            node = Node(node_id=nid,
                        address=node_configs[str(nid)]['IP'],
                        benchmarks=node_configs[str(nid)]['benchmarks'])
            node_pool.append(node)

    else:
        for nid in range(num_nodes):
            benchmarks = {'ram':random.choice(ram_variants), 'bandwidth':random.choice(bandwidth_variants)}
            random_ip_address = '.'.join(str(np.random.randint(0, 255)) for _ in range(4))
            # print(nid, random_ip_address, benchmarks)
            node = Node(node_id=nid,
                        address=random_ip_address,
                        benchmarks=benchmarks)
            node_pool.append(node)
    return node_pool

def cluster_formation(full_model_size, node_pool):
    prelim_clusters = genetic_algorithm(node_pool, full_model_size)
    prelim_clusters = dict(sorted(prelim_clusters.items()))
    prelim_clusters = {new_key: prelim_clusters[old_key] for new_key, old_key in enumerate(prelim_clusters.keys())}
    # print(prelim_clusters)
    clusters = [Cluster(cid) for cid in prelim_clusters.keys()]

    # assigning clusters to nodes based on genetic algorithm output
    for cid, nodes in prelim_clusters.items():
        for node in nodes:
            clusters[cid].add_node(node)

    for cluster in clusters:
        calculate_split_percentages(cluster=cluster, full_model_size=full_model_size)
        calculate_cluster_power(cluster=cluster)
        # cluster.state_dict = state_dict
    return clusters

def round_percentages(percentages):
    # Uses the "largest remainder" method, also known as the Hare-Niemeyer method
    integer_parts = [int(p) for p in percentages]
    remainders = [p - int_part for p, int_part in zip(percentages, integer_parts)]
    total = sum(integer_parts)
    remainder = 100 - total
    while remainder > 0:
        index = remainders.index(max(remainders))
        integer_parts[index] += 1
        remainders[index] = 0
        remainder -= 1
    return integer_parts

def calculate_cluster_power(cluster):
    total_ram = 0
    total_speed = 0
    for node in cluster.nodes.values():
        total_ram += node.benchmarks['ram']
        total_speed += node.benchmarks['ram'] / node.benchmarks['bandwidth']
    cluster.total_ram = total_ram
    cluster.total_speed = int(total_speed)


def calculate_split_percentages(cluster, full_model_size):
    rams = cluster.rams
    if sum(rams) < full_model_size:
        raise ValueError("The sum of the cluster rams ({}) does not exceed full_model_size ({}).".format(sum(rams), full_model_size))
    total_sum = sum(rams)
    percentages = [(ram / total_sum) * 100 for ram in rams]
    rounded_percentages = round_percentages(percentages)

    splits = [full_model_size * p // 100 for p in rounded_percentages]
    remainder = full_model_size - sum(splits)
    for i in range(remainder):
        splits[i % len(splits)] += 1

    cluster.splits = splits
    cluster.assign_split_quotas_to_nodes()
    
def view_individual_cluster_details(cluster_pool):
    print('\n')
    for cluster in cluster_pool:
        print('Cluster_ID: {}  | splits: {} | RAM: {} | Speed: {}'.format(cluster.cluster_id,  
                                                            cluster.splits,
                                                            cluster.total_ram,
                                                            cluster.total_speed
                                                            ))
        

def assign_connection_targets(cluster_pool):
    temp_splits = [cluster.splits for cluster in cluster_pool]
    splits = copy.deepcopy(temp_splits)
    copy_splits = copy.deepcopy(splits)

    for cl in range(len(copy_splits)):
        if cl == len(copy_splits) - 1:
            continuous_mapping, _ = representation_converter(cluster_split=copy_splits[0],target_split=copy_splits[cl])
        else:
            continuous_mapping, _ = representation_converter(cluster_split=copy_splits[cl + 1],target_split=copy_splits[cl])
        cluster_pool[cl].inter_cluster_node_address_mappings = continuous_mapping
                
    for cl in range(len(cluster_pool)):
        current_cluster = cluster_pool[cl]
        if cl == len(cluster_pool) - 1:
            next_cluster = cluster_pool[0]
        else:
            next_cluster = cluster_pool[cl + 1]
        
        address_param_mapping_list = []
        for nid_param_mapping in current_cluster.inter_cluster_node_address_mappings:
            address_param_mapping = copy.deepcopy(nid_param_mapping)
            for cid, param_idx in nid_param_mapping.items():
                actual_nid = list(next_cluster.nodes.keys())[cid]
                del address_param_mapping[cid]
                address_param_mapping[next_cluster.nodes[actual_nid].ip_address] = param_idx
            address_param_mapping_list.append(address_param_mapping)
        current_cluster.inter_cluster_node_address_mappings = address_param_mapping_list

    for cl in range(len(cluster_pool)):
        current_cluster = cluster_pool[cl]
        cluster_ip_map = current_cluster.inter_cluster_node_address_mappings
        
        for nid, node in current_cluster.nodes.items():            
            node.next_cluster_target_node_ip_to_param_mapping = cluster_ip_map[list(current_cluster.nodes.keys()).index(nid)]

        cluster_named_inter_cluster_node_address_mappings_list = []
        for nid, node in current_cluster.nodes.items():            
            for key, val in node.next_cluster_target_node_ip_to_param_mapping.items():
                node.next_cluster_target_node_ip_to_named_param_mapping[key] = current_cluster.state_dict[val] 
            cluster_named_inter_cluster_node_address_mappings_list.append(node.next_cluster_target_node_ip_to_named_param_mapping)

        current_cluster.named_inter_cluster_node_address_mappings = cluster_named_inter_cluster_node_address_mappings_list
        
    return True

def representation_converter(cluster_split, target_split):    
    # out = []
    # total_elements = sum(cluster_split)
    cluster_index = 0 
    cluster_count = cluster_split[0]
    r = {}
    continuous_result = []
    local_result = []
    start = 0
    cumulative_size = 0
    for size in target_split:
        end = start + size
        encoded_chunk = [-1] * len(cluster_split)
        for i in range(start, end):
            if cluster_count == 0:
                cluster_index += 1
                cluster_count = cluster_split[cluster_index]
            if encoded_chunk[cluster_index] == -1:
                encoded_chunk[cluster_index] = i - start
            cluster_count -= 1
        # out.append(encoded_chunk)
        local_r = {}
        continuous_r = {}
        for i in range(len(encoded_chunk)):
            if encoded_chunk[i] != -1:
                continuous_r[i] = encoded_chunk[i] + cumulative_size
                local_r[i] = encoded_chunk[i]
        cumulative_size += size
        continuous_result.append(continuous_r)
        local_result.append(local_r)
        start = end

    return continuous_result, local_result
    # return out


class CustomTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        
        if hasattr(m, '_is_leaf_module') and m._is_leaf_module:
            return True

        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)


# def split_model(model, n_splits=3):
#     custom_tracer = CustomTracer()
#     split_policy = split_into_equal_size(n_splits)
#     pipe = Pipe.from_tracing(model, tracer=custom_tracer, split_policy=split_policy)
#     return pipe

def split_model_on_proportions(model, proportions=[], example_args=None, example_kwargs=None):
    traced = Pipe._trace_with_export(model, example_args=example_args, example_kwargs=example_kwargs)#torch.jit.trace(model, example_inputs=input_ids)
    split_policy = split_on_proportions(proportions)
    traced = split_policy(traced)
    pipe = Pipe._from_traced(model, traced)
    return pipe

def get_arg_index(name, submod_args):
    for i in range(len(submod_args)):
        if submod_args[i].name == name:
            return i
    return -1


def split_model_equal(model=None, num_splits=None, proportions=None, example_args=(), example_kwargs={}, cluster_path=None, node_paths=None, model_input_node=None):

    # pipe = split_model(model, num_splits)
    pipe = split_model_on_proportions(model, proportions=proportions, example_args=example_args, example_kwargs=example_kwargs)
    compiled_input_dict = {}
    compiled_output_dict = {'model_inputs':{}}
    for node in pipe.split_gm.graph.nodes:
        # print(node.name, node.args)
        if 'submod' in node.name:
            input_dict = {}
            compiled_output_dict[node.name] = {}
            if node.name == 'submod_0':
                submod_0_args = node.args
                for i in range(len(submod_0_args)):
                    compiled_output_dict['model_inputs'][i] = {}
            else:
                if len(node.args) > 0:
                    for i in range(len(node.args)):
                        input_dict[i] = {}
                arg_index = 0
                for arg in node.args:
                    if 'submod' in arg.name:
                        input_dict[arg_index][arg.name] = 'placeholder:tensor'

                        if compiled_output_dict[arg.name].get(arg_index, None) is not None:
                            compiled_output_dict[arg.name][arg_index]['target'].append(node.name)   
                        else:
                            compiled_output_dict[arg.name][arg_index] = {'target' : [node.name]}


                    elif 'getitem' in arg.name:
                        inner_arg = arg.args             
                        input_dict[arg_index][inner_arg[0].name] = inner_arg[1]

                        if compiled_output_dict[inner_arg[0].name].get(inner_arg[1], None) is not None:
                            compiled_output_dict[inner_arg[0].name][inner_arg[1]]['target'].append(node.name)   
                        else:
                            compiled_output_dict[inner_arg[0].name][inner_arg[1]] = {'target' : [node.name]}

                    
                    else:
                        index = get_arg_index(arg.name, submod_0_args)
                        input_dict[arg_index]['model_inputs'] = index

                        if compiled_output_dict['model_inputs'][index].get('target', None) is not None:
                            compiled_output_dict['model_inputs'][index]['target'].append(node.name)
                        else:
                            compiled_output_dict['model_inputs'][index]['target'] = [node.name]
                    arg_index += 1
            compiled_input_dict[node.name] = input_dict

    for key, value in compiled_output_dict.items():
        if key == 'model_inputs':
            with open('{}/{}/{}.pkl'.format(cluster_path, model_input_node, key), 'wb') as file:
                pickle.dump(value,file)
        else:
            # l = key.split('_')[:-1]
            k = key.split('_')[-1]
            with open('{}/{}/{}_output.pkl'.format(cluster_path, node_paths[int(k)],key), 'wb') as file:
                pickle.dump(value,file)

    for key, value in compiled_input_dict.items():        
        # l = key.split('_')[:-1]
        k = key.split('_')[-1]
        # print('key in compiled dict: ', l, k)
        with open('{}/{}/{}_input.pkl'.format(cluster_path, node_paths[int(k)],key), 'wb') as file:
            pickle.dump(value,file)

    print('\nSubmodels are Saved in: ')        
    for key, val in pipe.split_gm._modules.items():
        script = torch.jit.script(val)
        k = key.split('_')[-1]
        print('{}/{}/{}.pt'.format(cluster_path, node_paths[int(k)], key))
        script.save('{}/{}/submod.pt'.format(cluster_path, node_paths[int(k)]))

def delete_all_folders(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

def clusterize(model=None, proportions=[], example_args = (), example_kwargs = {}):
    """Takes the complete deep learning model and forms clusters from a pool of compute nodes defined in ```node_data/node_configs.json``` file. Automates the whole process of address sharing across nodes, reduction ring formation and seamlessly stores the results as node metadata json files for each node in ```node_data/nodes/``` folder. These metadata files are later used by ```ravnest.node.Node``` class to load all relevant attributes pertaining to a node.

    :param model: Pytorch Model, defaults to None
    :raises ValueError: If the sum of the node RAMs in a cluster does not exceed full model's size.
    """        
    random.seed(42)
    np.random.seed(42)

    path = 'node_data/'
    delete_all_folders(path)

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
                        # num_splits=len(cluster.nodes),
                        proportions=proportions, 
                        example_args=example_args,
                        example_kwargs=example_kwargs,
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

    print('\nNo. rings: ', max_l)
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
        
    # max_ring_size_value = max(max_ring_size.values())
    max_ring_size_value = len(cluster_pool)
    print('Max ring size: ', max_ring_size)

    for cl in range(len(cluster_pool)):
        cluster = cluster_pool[cl]
        if cl == len(cluster_pool) - 1:
            next_cluster = cluster_pool[0]
        else:
            next_cluster = cluster_pool[cl + 1]

        for nid, node in cluster.nodes.items():
            current_address = None
            node.cluster_length = len(cluster.nodes)
            for k in node.trainable_param_keys:
                for n_nid, n_node in next_cluster.nodes.items():
                    if k in n_node.trainable_param_keys:
                        if current_address != n_node.address:
                            node.address_to_param[n_node.address] = k
                            current_address = n_node.address

    print('\n------------------------------------------------')
    for node in node_pool:
        print(node)
        node_meta = {}
        node_meta['node_id'] = node.node_id
        node_meta['local_host'] = node.address.split(':')[0]
        node_meta['local_port'] = int(node.address.split(':')[1])
        node_meta['template_path'] = 'node_data/cluster_{}/{}/'.format(node.cluster_id, node.address)
        node_meta['rank'] = node.cluster_id
        node_meta['ring_size'] = max_ring_size_value
        node_meta['cluster_length'] = node.cluster_length
        node_meta['param_addresses'] = node.address_to_param,
        node_meta['ring_ids'] = {int(key): value for key, value in node.ring_ids.items()}
        node_meta['forward_target_host'] = node.forward_target_host
        node_meta['forward_target_port'] = int(node.forward_target_port) if node.forward_target_port is not None else None
        node_meta['backward_target_host'] = node.backward_target_host
        node_meta['backward_target_port'] = int(node.backward_target_port) if node.backward_target_port is not None else None

        with open('node_data/nodes/node_{}.json'.format(node.node_id), 'w') as fp:
            json.dump(node_meta, fp)
    print('\nClusters Formed Successfully!')