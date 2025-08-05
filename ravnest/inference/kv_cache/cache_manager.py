'''
Based on https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/inference/kv_cache/kvcache_manager.py
'''

import torch
from typing import List, Tuple
from .kv_block import CacheBlock

class CacheManager():

    def __init__(self, device=None, dtype=None, num_shard_layers=None, model_config=None, 
                max_batch_size=None, max_seq_length_during_gen=None,
                block_size=None):
        
        self.device = device
        self.kv_cache_dtype = dtype

        print('kv cache dtype: ', self.kv_cache_dtype)

        self.elem_size_in_bytes = torch.tensor([], dtype=self.kv_cache_dtype).element_size()

        '''
        CHANGE NUM LAYERS FOR EACH STAGE
        '''
        self.num_layers = num_shard_layers #model_config.num_hidden_layers
        self.head_num = model_config.num_attention_heads
        self.head_size = model_config.hidden_size // self.head_num

        if hasattr(model_config, "num_key_value_heads"):
            self.kv_head_num = model_config.num_key_value_heads
        else:
            self.kv_head_num = self.head_num

        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length_during_gen

        self.block_size = block_size
        self.max_blocks_per_sequence = (self.max_seq_length + self.block_size - 1) // self.block_size
        self.num_blocks = self.max_blocks_per_sequence * self.max_batch_size

        print('Block size: ', self.block_size)
        print('Max seq length during gen: ', self.max_seq_length)
        print('Max blocks per seq: ', self.max_blocks_per_sequence)
        print('Num blocks: ', self.num_blocks)

        x = 16 // torch.tensor([], dtype=self.kv_cache_dtype).element_size()
        kalloc_shape = (self.num_blocks, self.kv_head_num, self.head_size // x, self.block_size, x)
        valloc_shape = (self.num_blocks, self.kv_head_num, self.block_size, self.head_size)
        self._kv_caches = self._init_device_caches(kalloc_shape, valloc_shape)

        self.total_physical_cache_size_in_bytes = (
            self.elem_size_in_bytes
            * self.num_layers
            * 2
            * self.num_blocks
            * self.block_size
            * self.kv_head_num
            * self.head_size
        )

        print(f'Allocated {self.total_physical_cache_size_in_bytes / (1024**3):.2f} GB of KV cache on device {self.device}.')

        # Logical cache blocks allocation
        self._available_blocks = self.num_blocks
        self._cache_blocks = tuple(self._init_logical_caches())
        # block availablity state 0->allocated, 1->free
        self._block_states = torch.ones((self.num_blocks,), dtype=torch.bool)
        self._block_states_cum = torch.zeros(size=(self.num_blocks + 1,), dtype=torch.int64)
        self._block_finder = torch.zeros((self.num_blocks,), dtype=torch.int64)

    def _init_device_caches(
        self, kalloc_shape: Tuple[int, ...], valloc_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the physical cache on the device.

        For each layer of the model, we allocate two tensors for key and value respectively,
        with shape of [num_blocks, num_kv_heads, block_size, head_size]
        """
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            k_cache.append(torch.zeros(kalloc_shape, dtype=self.kv_cache_dtype, device=self.device))
            v_cache.append(torch.zeros(valloc_shape, dtype=self.kv_cache_dtype, device=self.device))
            print('Kalloc shape, valloc shape for layer: ', kalloc_shape, valloc_shape)
        return k_cache, v_cache

    def _init_logical_caches(self):
        """Initialize the logical cache blocks.

        NOTE This function should be called only after the physical caches have been allocated.
        The data pointers of physical caches will be binded to each logical cache block.
        """
        assert self._kv_caches is not None and len(self._kv_caches[0]) > 0
        blocks = []
        physical_block_size = self.elem_size_in_bytes * self.block_size * self.kv_head_num * self.head_size
        k_ptrs = [
            self._kv_caches[0][layer_idx].data_ptr() - physical_block_size for layer_idx in range(self.num_layers)
        ]
        v_ptrs = [
            self._kv_caches[1][layer_idx].data_ptr() - physical_block_size for layer_idx in range(self.num_layers)
        ]
        for i in range(self.num_blocks):
            k_ptrs = [first_block_ptr + physical_block_size for first_block_ptr in k_ptrs]
            v_ptrs = [first_block_ptr + physical_block_size for first_block_ptr in v_ptrs]
            cache_block = CacheBlock(i, self.block_size, self.elem_size_in_bytes, k_ptrs, v_ptrs)
            blocks.append(cache_block)
        return blocks

    def get_kv_cache_shapes(self):
        return (2, self.num_blocks, self.kv_head_num, self.block_size, self.head_size)

    def get_kv_caches(self):
        return self._kv_caches

    def _allocate_on_block(self, block: CacheBlock, space_asked: int) -> int:
        """Allocate a specific size of space on a provided cache block.

        Returns:
            The remaining space required to be allocated (in other blocks).
        """
        assert block.available_space > 0, f"Found no available space left in the chosen block {block}."
        space_to_allocate = min(block.available_space, space_asked)
        block.allocate(space_to_allocate)
        return space_asked - space_to_allocate

    def free_block_table(self, block_table: torch.Tensor) -> None:
        """Free the logical cache blocks for **a single sequence**."""
        assert block_table.dim() == 1
        for i, global_block_id in enumerate(block_table.tolist()):
            if global_block_id < 0:
                return
            block: CacheBlock = self._cache_blocks[global_block_id]
            block.remove_ref()
            if not block.has_ref():
                block.allocated_size = 0
                self._available_blocks += 1
                self._block_states[global_block_id] = 1
                # reset the block id in the block table (if we maintain a 2D tensors as block tables in Engine)
                block_table[i] = -1

    '''
    CHANGE FOR PADDING
    '''
    def allocate_cache_blocks_to_tables(self, block_tables, context_lengths):

        assert block_tables.size(0) == context_lengths.size(0)

        blocks_required_per_seq = (context_lengths + self.block_size - 1) // self.block_size
        num_blocks_required = torch.sum(blocks_required_per_seq).item()

        if num_blocks_required > self._available_blocks:
            print(
                f"Lacking blocks to allocate. Available blocks {self._available_blocks}; blocks asked {num_blocks_required}."
            )
            return

        bsz = block_tables.size(0)
        torch.cumsum(self._block_states, dim=-1, out=self._block_states_cum[1:])
        torch.subtract(
            self._block_states_cum[num_blocks_required:],
            self._block_states_cum[:-num_blocks_required],
            out=self._block_finder[num_blocks_required - 1 :],
        )
        end_indexes = torch.nonzero(self._block_finder == num_blocks_required, as_tuple=False).view(-1)
        if end_indexes.numel() > 0:
            # contiguous cache exists
            end_idx = end_indexes[0].item() + 1  # open interval
            start_idx = end_idx - num_blocks_required  # closed interval
            alloc_block_ids = torch.arange(start_idx, end_idx)
            for i in range(bsz):
                curr_required = blocks_required_per_seq[i]
                block_tables[i, :curr_required] = torch.arange(
                    start_idx, start_idx + curr_required, device=block_tables.device
                )
                start_idx += curr_required
        else:
            # non-contiguous cache
            available_block_ids = torch.nonzero(self._block_states > 0).view(-1)
            alloc_block_ids = available_block_ids[:num_blocks_required]
            alloc_block_ids = alloc_block_ids.to(dtype=block_tables.dtype, device=block_tables.device)
            start_idx = 0
            for i in range(bsz):
                curr_required = blocks_required_per_seq[i]
                block_tables[i, :curr_required] = alloc_block_ids[start_idx, start_idx + curr_required]
                start_idx += curr_required

        # Update cache blocks
        self._block_states[alloc_block_ids] = 0
        self._available_blocks -= num_blocks_required
        last_block_locs = torch.cumsum(blocks_required_per_seq, dim=0) - 1
        last_block_locs = last_block_locs.to(device=alloc_block_ids.device)

        for i, block_id in enumerate(alloc_block_ids[last_block_locs]):
            block: CacheBlock = self._cache_blocks[block_id]
            block.add_ref()
            self._allocate_on_block(
                block,
                (
                    block.block_size
                    if context_lengths[i] % block.block_size == 0
                    else context_lengths[i].item() % block.block_size
                ),
            )
        for block_id in alloc_block_ids:
            if block_id in alloc_block_ids[last_block_locs]:
                continue
            block: CacheBlock = self._cache_blocks[block_id]
            block.add_ref()
            self._allocate_on_block(block, block.block_size)

    def allocate_cache_blocks_to_tables_for_new_tokens(self, block_tables, context_lengths):
        """Allocate logical cache blocks for a batch of sequences during decoding stage.

        Usage:
            allocate_context_from_block_tables
            model forward (block tables & context lengths passed)
            update context lengths
            allocate_tokens_from_block_tables
            model forward
            update context lengths
            allocate_tokens_from_block_tables
            model forward
            update context lengths
            ...

        Args:
            block_tables (torch.Tensor): [bsz, max_blocks_per_sequence]
            context_lengths (torch.Tensor): [bsz]

        Returns:
            List[int]: list of sequence uid to be recycled
        """
        assert block_tables.dim() == 2
        assert context_lengths.dim() == 1

        bsz = block_tables.size(0) #if bsz is None else bsz

        alloc_local_block_indexes = (context_lengths[:bsz]) // self.block_size
        block_global_ids = block_tables[torch.arange(0, bsz), alloc_local_block_indexes]
        seqs_to_recycle = []
        new_blocks_required = torch.sum(block_global_ids < 0).item()
        seqs_req_new_blocks = torch.nonzero(block_global_ids < 0).squeeze()

        if new_blocks_required > 0:
            if new_blocks_required > self._available_blocks:
                # TODO might want to revise the logic here
                # Process the first (_available_blocks) sequences that require new blocks
                # Put the rest of the sequences back to recycled
                seqs_req_new_blocks, seqs_to_recycle = (
                    seqs_req_new_blocks[: self._available_blocks],
                    seqs_req_new_blocks[self._available_blocks :],
                )
                for seq_id in seqs_to_recycle:
                    self.free_block_table(block_tables[seq_id])
                new_blocks_required = self._available_blocks

            # NOTE might want to alloc contiguous logic
            free_block_ids = torch.nonzero(self._block_states > 0).view(-1)
            alloc_block_ids = free_block_ids[:new_blocks_required].to(
                dtype=block_tables.dtype, device=block_tables.device
            )

            for block_id in alloc_block_ids:
                block: CacheBlock = self._cache_blocks[block_id]
                block.add_ref()
                self._block_states[block_id] = 0
                self._available_blocks -= 1
            block_tables[seqs_req_new_blocks, alloc_local_block_indexes[seqs_req_new_blocks]] = alloc_block_ids
            block_global_ids = block_tables[torch.arange(0, bsz), alloc_local_block_indexes]

        for block_id in block_global_ids:
            self._allocate_on_block(self._cache_blocks[block_id], 1)

        return seqs_to_recycle