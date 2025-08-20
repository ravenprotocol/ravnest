import torch
from .cache_manager import CacheManager
from .kv_block import CacheBlock

class PagedAttentionEngine():
    def __init__(self, device, dtype=None, num_shard_layers = None, model_config= None,
                max_batch_size=None, max_seq_length_during_gen=None, 
                block_size=None):

        self.device = device
        self.dtype = dtype
        self.model_config = model_config
        self.num_shard_layers = num_shard_layers
        self.max_batch_size = max_batch_size
        self.max_seq_length_during_gen = max_seq_length_during_gen
        self.block_size = block_size

        max_blocks_per_seq = (self.max_seq_length_during_gen + self.block_size - 1) // self.block_size
        self._block_tables = torch.full((self.max_batch_size, max_blocks_per_seq), -1, dtype=torch.int32)
        self._block_tables_helper = torch.full_like(self._block_tables, -1)

        self.cache_manager = CacheManager(device=self.device, 
                                          dtype=self.dtype, 
                                          num_shard_layers=self.num_shard_layers,
                                          model_config=self.model_config,
                                          max_batch_size=self.max_batch_size, 
                                          max_seq_length_during_gen=self.max_seq_length_during_gen,
                                          block_size=self.block_size
                                        )

    def allocate_block_tables_for_batch(self, batch, seq_lengths):
        block_tables = self._block_tables[:len(batch)]
        self.cache_manager.allocate_cache_blocks_to_tables(block_tables, seq_lengths)
        # print('Block tables after prefill allocation: ', block_tables)
        return block_tables

    def allocate_block_tables_for_new_tokens(self, batch, seq_lengths):
        block_tables = self._block_tables[:len(batch)]
        seqs_to_be_recycled = self.cache_manager.allocate_cache_blocks_to_tables_for_new_tokens(block_tables, seq_lengths)
        # print('Block tables after new tokens allocation: ', block_tables)
        return block_tables

    def get_kv_cache_shapes(self):
        return self.cache_manager.get_kv_cache_shapes()

    def get_kv_caches(self):
        return self.cache_manager.get_kv_caches()

    def get_block_tables(self, bs):
        block_tables = self._block_tables[:bs]
        return block_tables.to(self.device)
