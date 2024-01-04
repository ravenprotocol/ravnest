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

# model_splits = split_models(model_data=model_data, splits=splits)
# print(model_splits[0][2].shape) # [cluster_id][node_id]

original_clusters = copy.deepcopy(splits)
clusters = copy.deepcopy(splits)

print('\nInput: ')
for cl in range(len(clusters)):
    print(clusters[cl])

max_rings = max(len(cluster) for cluster in clusters)
print('\nMax rings: ', max_rings)


for ring_id in range(max_rings):

    min_node_size = min([cluster[ring_id] for cluster in clusters])

    for cl in range(len(clusters)):

        if clusters[cl][ring_id] > min_node_size:
            diff = clusters[cl][ring_id] - min_node_size
            my_split = splits[cl]
            if ring_id < len(splits[cl]) - 1:
                my_split[ring_id] = my_split[ring_id] - diff
                my_split[ring_id + 1] = my_split[ring_id + 1] + diff
            elif ring_id == len(splits[cl]) - 1:
                t = my_split[ring_id]
                my_split[ring_id] = t - diff
                my_split.append(diff)

            splits[cl] = my_split
            clusters[cl] = my_split

localized_rings = []
continuous_rings = []

for cl in range(len(clusters)):
    continuous_r, local_r = get_id_representation(clusters[cl], original_clusters[cl])
    localized_rings.append(local_r)
    continuous_rings.append(continuous_r)

print('\nLocalized Output: ')
for local_r in localized_rings:
    print(local_r)

print('\nContinuous Output: ')
for continuous_r in continuous_rings:
    print(continuous_r)
