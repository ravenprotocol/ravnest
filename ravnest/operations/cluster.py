class Cluster:
    def __init__(self, cid) -> None:
        self.cid = cid
        self.nodes = {}
        self.size = 0
        self.rams = []
        self.total_ram = 0
        self.total_speed = 0
        self.splits = None

    def add_node(self, node):
        node.set_cluster_id(self.cid)
        self.nodes[node.node_id] = node
        self.rams.append(node.benchmarks['ram'])
        self.size += 1

    def set_ringwise_params(self):
        all_param_to_ring = {}
        for nid, node in self.nodes.items():
            for param in node.trainable_param_keys:
                all_param_to_ring[param] = list(node.ring_ids.keys())[0]

        self.all_param_to_ring = all_param_to_ring

    def assign_split_quotas_to_nodes(self):
        q = 0
        for id, node in self.nodes.items():
            node.split_quota = self.splits[q]
            q += 1
    
    def __repr__(self) -> str:
        return "\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nCluster({})\nNodes:{} \nSplits({})".format(self.cid,
                                                list(self.nodes.keys()),
                                                self.splits
                                      )