from cluster_node_operations.utils import *
import random
from pprint import pprint


# Simulating Distributed Environment Parameters
ram_variants = [2,4,8,16] #[2,4,8,16,32,64,128]
bandwidth_variants = [random.randint(1,40) for _ in range(10)]
num_nodes = 20
full_model_size = 25 #gb

node_pool, cluster_pool, formed_rings, assigned_connection_targets = configure_clusters(max_attempts=5, 
                                                                                            num_nodes=num_nodes,
                                                                                            full_model_size=full_model_size, 
                                                                                            ram_variants=ram_variants, 
                                                                                            bandwidth_variants=bandwidth_variants)

view_individual_cluster_details(cluster_pool=cluster_pool)

print('\nCluster Pool:')
pprint(cluster_pool)

print('\nNode Pool:')
pprint(node_pool)