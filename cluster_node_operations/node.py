import torch

class Node:
    def __init__(self, node_id, cluster=None, ip_address=None, benchmarks=None, submodel=None):
        self.node_id = node_id
        self.cluster = cluster
        self.ip_address = ip_address
        self.benchmarks = benchmarks
        self.submodel = submodel
        self.split_quota = None
        self.ring_id_to_param_mapping = None
        self.ring_id_to_named_param_mapping = {}
        self.next_cluster_target_node_ip_to_param_mapping = None
        self.next_cluster_target_node_ip_to_named_param_mapping = {}
        self.submodel = None

    def set_submodel(self):
        self.submodel = torch.jit.load('node_data/cluster_{}/{}/submod.pt'.format(self.cluster.cluster_id,
                                                                self.ip_address))
        
    def set_ring_id_to_named_param_mapping(self):
        for ring_id, param in self.ring_id_to_param_mapping.items():
            self.ring_id_to_named_param_mapping[ring_id] = list(self.submodel.state_dict().keys())[param]

    def set_cluster(self, cluster):
        self.cluster = cluster

    def __repr__(self):
        return "\n\nNode({}, Cluster({})) \nself.IP({}) \nIPAdd2NamedParamMapping({}) \nRID2NamedParamMapping({})".format(self.node_id,
                                                                                                        self.cluster.cluster_id if self.cluster else 'None', 
                                                                                                        self.ip_address,                                            
                                                                                                        # self.next_cluster_target_node_ip_to_param_mapping,
                                                                                                        self.next_cluster_target_node_ip_to_named_param_mapping,
                                                                                                        # self.ring_id_to_param_mapping,
                                                                                                        self.ring_id_to_named_param_mapping)
