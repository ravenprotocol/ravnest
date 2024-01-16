class Cluster:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.nodes = {}
        self.rams = []
        self.splits = None 
        self.total_ram = 0
        self.total_speed = 0
        self.ring_id_mapping = None
        self.ring_id_named_mapping = None
        self.state_dict = None
        self.inter_cluster_node_address_mappings = None
        self.named_inter_cluster_node_address_mappings = None

    def add_node(self, node):
        if node.node_id in self.nodes:
            raise ValueError("Node ID {} already exists in this cluster".format(node.node_id))
        self.nodes[node.node_id] = node
        self.rams.append(node.benchmarks['ram'])
        node.set_cluster(self)

    def get_node(self, node_id):
        if self.nodes.get(node_id, None) is None:
            return False
        return self.nodes[node_id]

    def assign_split_quotas_to_nodes(self):
        q = 0
        for id, node in self.nodes.items():
            node.split_quota = self.splits[q]
            q += 1

    def __repr__(self):

        return "\nxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nCluster({})\nNodes:{}  \nICAM:{} \nNamedICAM:{} \nRIDM:{}".format(self.cluster_id,
                                                list(self.nodes.keys()),
                                                self.inter_cluster_node_address_mappings,
                                                self.named_inter_cluster_node_address_mappings,
                                                self.ring_id_mapping
                                      )
                                      