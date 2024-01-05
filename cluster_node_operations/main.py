from utils import *
import random
from pprint import pprint


# Simulating Distributed Environment Parameters
ram_variants = [2,4,8,16] #[2,4,8,16,32,64,128]
bandwidth_variants = [random.randint(1,40) for _ in range(10)]
num_nodes = 20
full_model_size = 25 #gb


node_pool = spawn_node_pool(num_nodes=num_nodes, ram_variants=ram_variants, bandwidth_variants=bandwidth_variants)
cluster_pool = cluster_formation(full_model_size=full_model_size, node_pool=node_pool)


form_rings(cluster_pool=cluster_pool)

whom_to_connect_with(cluster_pool=cluster_pool)

view_individual_cluster_details(cluster_pool=cluster_pool)

print('\nCluster Pool:')
pprint(cluster_pool)

print('\nNode Pool:')
pprint(node_pool)