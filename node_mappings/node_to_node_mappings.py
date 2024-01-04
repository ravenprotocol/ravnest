from utils import get_id_representation
import numpy as np
import copy

c0_data = np.random.random((10,20,25))   # entire model for cluster
c1_data = np.random.random((10,20,25))   # entire model for cluster
c2_data = np.random.random((10,20,25))   # entire model for cluster

model_data = [c0_data, c1_data, c2_data]


c0_split = [4,3,3]
c1_split = [2,4,3,1]
c2_split = [2,5,1,1,1]

splits = [c0_split, c1_split, c2_split]

clusters = copy.deepcopy(splits)

print(clusters)

for cl in range(len(clusters)):
    if cl == len(clusters) - 1:
        continuous_ids, _ = get_id_representation(clusters[0],clusters[cl])
    else:
        continuous_ids, _ = get_id_representation(clusters[cl + 1],clusters[cl])
    print('\nCluster {}: '.format(cl), continuous_ids)
    