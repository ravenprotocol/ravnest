import torch
from cluster_node_operations.misc import get_trainable_parameters

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
        # self.submodel = None

    def set_submodel(self):
        self.submodel = torch.jit.load('node_data/cluster_{}/{}/submod.pt'.format(self.cluster.cluster_id,
                                                                self.ip_address))

    def set_next_cluster_target_node_ip_to_named_param_mapping(self):
        print('self.nid: ',self.node_id ,self.next_cluster_target_node_ip_to_param_mapping, len(list(get_trainable_parameters(self.submodel).keys())))
        i=self.cluster.state_dict.index(list(get_trainable_parameters(self.submodel).keys())[0])
        for ip, param in self.next_cluster_target_node_ip_to_param_mapping.items():
            # self.next_cluster_target_node_ip_to_named_param_mapping[ip] = list(self.submodel.state_dict().keys())[param]
            self.next_cluster_target_node_ip_to_named_param_mapping[ip] = self.cluster.state_dict[i+param]            
        print(self.next_cluster_target_node_ip_to_named_param_mapping)

    def set_ring_id_to_named_param_mapping(self):
        for ring_id, param in self.ring_id_to_param_mapping.items():
            self.ring_id_to_named_param_mapping[ring_id] = self.cluster.state_dict[param]

    def set_cluster(self, cluster):
        self.cluster = cluster

    def __repr__(self):
        return "\n\nNode({}, Cluster({})) \nself.IP({}) \nIPAdd2ParamMap({}) \nIPAdd2NamedParamMapping({}) \nRID2Param({}) \nRID2NamedParamMapping({})".format(self.node_id,
                                                                                                        self.cluster.cluster_id if self.cluster else 'None', 
                                                                                                        self.ip_address,                                            
                                                                                                        self.next_cluster_target_node_ip_to_param_mapping,
                                                                                                        self.next_cluster_target_node_ip_to_named_param_mapping,
                                                                                                        self.ring_id_to_param_mapping,
                                                                                                        self.ring_id_to_named_param_mapping)
