import numpy as np

def split_models(model_data, splits):
    model_splits = []
    for s in range(len(splits)):
        model_splits.append(np.split(model_data[s], np.cumsum(splits[s])[:-1], axis=0))
    return model_splits


def get_id_representation(cluster, split):    
    # out = []
    # total_elements = sum(cluster)
    cluster_index = 0 
    cluster_count = cluster[0]
    r = {}
    continuous_result = []
    local_result = []
    start = 0
    cumulative_size = 0
    for size in split:
        
        end = start + size
        encoded_chunk = [-1] * len(cluster)
        for i in range(start, end):
            if cluster_count == 0:
                cluster_index += 1
                cluster_count = cluster[cluster_index]
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
