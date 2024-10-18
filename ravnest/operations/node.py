import torch

class Node:
    def __init__(self, node_id=None, address=None, benchmarks=None) -> None:
        self.node_id = node_id
        self.ring_ids = {}
        self.address_to_param = {}
        self.address = address
        self.benchmarks = benchmarks
        self.split_quota = None
        self.cluster_id = None
        self.trainable_param_keys = []
        self.forward_target_host = None
        self.forward_target_port = None
        self.backward_target_host = None
        self.backward_target_port = None
        self.cluster_length = 1
        self.input_shape = None
        self.output_shape = None

    def set_submodel(self):
        self.submodel = torch.jit.load('node_data/cluster_{}/{}/submod.pt'.format(self.cluster_id, self.address))

    def set_cluster_id(self, cid):
        self.cluster_id = cid

    @torch.no_grad()
    def set_trainable_parameter_keys(self):
        for name, param in self.submodel.named_parameters():
            if param.requires_grad:
                self.trainable_param_keys.append(name)

    def __repr__(self):
        return "\n\nNode({}, Cluster({})) \nself.IP({}) \nRing IDs({}) \nAddress2Param({})".format(self.node_id,
                                                                self.cluster_id, 
                                                                self.address, self.ring_ids, self.address_to_param)

