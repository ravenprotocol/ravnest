from genetic import *
from cluster import Cluster
from node import Node
import numpy as np
import random
import copy

def spawn_node_pool(num_nodes, ram_variants, bandwidth_variants):
    node_pool = []
    for nid in range(num_nodes):
        benchmarks = {'ram':random.choice(ram_variants), 'bandwidth':random.choice(bandwidth_variants)}
        random_ip_address = '.'.join(str(np.random.randint(0, 255)) for _ in range(4))
        node = Node(node_id=nid,
                    ip_address=random_ip_address,
                    benchmarks=benchmarks)
        node_pool.append(node)
    return node_pool

def cluster_formation(full_model_size, node_pool):
    prelim_clusters = genetic_algorithm(node_pool, full_model_size)
    prelim_clusters = dict(sorted(prelim_clusters.items()))
    prelim_clusters = {new_key: prelim_clusters[old_key] for new_key, old_key in enumerate(prelim_clusters.keys())}

    clusters = [Cluster(cid) for cid in prelim_clusters.keys()]

    # assigning clusters to nodes based on genetic algorithm output
    for cid, nodes in prelim_clusters.items():
        for node in nodes:
            clusters[cid].add_node(node)

    for cluster in clusters:
        calculate_split_percentages(cluster=cluster, full_model_size=full_model_size)
        calculate_cluster_power(cluster=cluster)
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
        print('Cluster_ID: {}  | splits: {} | RAM: {} | Speed: {} | RingID Mapping: {}'.format(cluster.cluster_id,  
                                                            cluster.splits,
                                                            cluster.total_ram,
                                                            cluster.total_speed,
                                                            cluster.ring_id_mapping
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
        
    return 

def whom_to_connect_with(cluster_pool):
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

