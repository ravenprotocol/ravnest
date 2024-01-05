class Node:
    def __init__(self, node_id, cluster=None, ip_address=None, benchmarks=None, submodel=None):
        self.node_id = node_id
        self.cluster = cluster
        self.ip_address = ip_address
        self.benchmarks = benchmarks
        self.submodel = submodel
        self.split_quota = None
        self.ring_id_to_param_mapping = None
        self.next_cluster_target_node_ip_to_param_mapping = None

    def set_cluster(self, cluster):
        self.cluster = cluster

    def __repr__(self):
        return "Node({}, Cluster({}), self.IP({}), IP Address to Param Mapping({}))".format(self.node_id,
                                                            self.cluster.cluster_id if self.cluster else 'None', 
                                                            self.ip_address,                                            
                                                            self.next_cluster_target_node_ip_to_param_mapping)
        

        # return "Node({}, Cluster({}), IP Address({}), Benchmarks({}), Split_Quota({}), RingID to Param Mapping: {})".format(self.node_id,
        #                                                                       self.cluster.cluster_id if self.cluster else 'None', 
        #                                                                       self.ip_address, 
        #                                                                       self.benchmarks,
        #                                                                       self.split_quota,
        #                                                                       self.ring_id_to_param_mapping,
        #                                                                       self.next_cluster_target_node_ip_to_param_mapping
        #                                                                         )
