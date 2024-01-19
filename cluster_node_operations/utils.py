import torch
from torch import nn
import pickle
from pip._internal.operations.freeze import freeze

from torch.fx import Tracer
from pippy.IR import Pipe
from pippy import split_into_equal_size

from cluster_node_operations.genetic import *
from cluster_node_operations.cluster import Cluster
from cluster_node_operations.node import Node
import numpy as np
import random
import json
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

def configure_clusters(max_attempts=5, num_nodes=None, full_model_size=None, ram_variants=None, bandwidth_variants=None):
    for attempt in range(max_attempts):
        try:
            node_pool = spawn_node_pool(num_nodes=num_nodes, ram_variants=ram_variants, bandwidth_variants=bandwidth_variants)
            cluster_pool = cluster_formation(full_model_size=full_model_size, node_pool=node_pool)
            formed_rings = form_rings(cluster_pool=cluster_pool)
            assigned_connection_targets = assign_connection_targets(cluster_pool=cluster_pool)

            if assigned_connection_targets is not None:  # Check if the function converged
                return node_pool, cluster_pool, formed_rings, assigned_connection_targets
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            # Optional: Handle specific exceptions if needed
    print(f"Failed to converge after {max_attempts} attempts.")
    return None  # or handle this case as needed


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
        
def form_rings(cluster_pool):
    temp_splits = [cluster.splits for cluster in cluster_pool]
    splits = copy.deepcopy(temp_splits)
    original_splits = copy.deepcopy(splits)
    splits_copy = copy.deepcopy(splits)
    max_rings = max(len(split) for split in splits_copy)
    print('\nMax rings: ', max_rings)
    for ring_id in range(max_rings):
        min_node_size = min([split[ring_id] for split in splits_copy])
        for cl in range(len(splits_copy)):
            if splits_copy[cl][ring_id] > min_node_size:
                diff = splits_copy[cl][ring_id] - min_node_size
                my_split = splits[cl]
                if ring_id < len(splits[cl]) - 1:
                    my_split[ring_id] = my_split[ring_id] - diff
                    my_split[ring_id + 1] = my_split[ring_id + 1] + diff
                elif ring_id == len(splits[cl]) - 1:
                    t = my_split[ring_id]
                    my_split[ring_id] = t - diff
                    my_split.append(diff)
                splits[cl] = my_split
                splits_copy[cl] = my_split

    localized_rings = []
    continuous_rings = []

    for cl in range(len(splits_copy)):
        continuous_rep, local_rep = representation_converter(cluster_split=splits_copy[cl], target_split=original_splits[cl])
        localized_rings.append(local_rep)
        continuous_rings.append(continuous_rep)

    # print('\nLocalized Ring Output: ')
    # for local_rep in localized_rings:
    #     print(local_rep)

    # print('\nContinuous Ring Output: ')
    # for continuous_rep in continuous_rings:
    #     print(continuous_rep)

    for cl in range(len(cluster_pool)):
        cluster_pool[cl].ring_id_mapping = continuous_rings[cl]
        r = 0
        for nid, node in cluster_pool[cl].nodes.items():
            node.ring_id_to_param_mapping = continuous_rings[cl][r]
            r += 1
        
    # for cl in range(len(cluster_pool)):
    #     cluster_pool[cl].ring_id_mapping = localized_rings[cl]
    #     r = 0
    #     for nid, node in cluster_pool[cl].nodes.items():
    #         node.ring_id_to_param_mapping = localized_rings[cl][r]
    #         r += 1
        
    return 

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


def split_model(model, n_splits=3):
    custom_tracer = CustomTracer()
    split_policy = split_into_equal_size(n_splits)
    pipe = Pipe.from_tracing(model, tracer=custom_tracer, split_policy=split_policy)
    return pipe

def get_arg_index(name, submod_args):
    for i in range(len(submod_args)):
        if submod_args[i].name == name:
            return i
    return -1


def split_model_equal(model=None, num_splits=None, cluster_path=None, node_paths=None, model_input_node=None):

    pipe = split_model(model, num_splits)
    compiled_input_dict = {}
    compiled_output_dict = {'model_inputs':{}}
    for node in pipe.split_gm.graph.nodes:
        print(node.name, node.args)
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
            
    for key, val in pipe.split_gm._modules.items():
        script = torch.jit.script(val)
        k = key.split('_')[-1]
        print('{}/{}/{}.pt'.format(cluster_path, node_paths[int(k)], key))
        script.save('{}/{}/submod.pt'.format(cluster_path, node_paths[int(k)]))


def simple_assign_connection_targets(cluster_pool):
    temp_splits = [cluster.splits for cluster in cluster_pool]
    splits = copy.deepcopy(temp_splits)
    copy_splits = copy.deepcopy(splits)

    for cl in range(len(copy_splits)):
        if cl == len(copy_splits) - 1:
            continuous_mapping, _ = representation_converter(cluster_split=copy_splits[0],target_split=copy_splits[cl])
        else:
            continuous_mapping, _ = representation_converter(cluster_split=copy_splits[cl + 1],target_split=copy_splits[cl])
        cluster_pool[cl].inter_cluster_node_address_mappings = _#continuous_mapping
                
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