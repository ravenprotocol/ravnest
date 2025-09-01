'''
Based on https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/inference/modeling/layers/attention.py
'''
import torch
import torch.nn.functional as F

# def copy_to_cache(source, cache, lengths, block_tables, type: str = "prefill"):
#     """
#     Func: copy key/value into key/value cache.

#     Args:   key/value(source): shape [bsz,num_heads,seq_len,head_size]
#             cache: shape [num_blocks, num_kv_heads, block_size, head_size]
#             lengths: key/value lengths
#             block_tables
#     """
#     # print('Block tables shape: ', block_tables.shape, ' Cache shape: ', cache.shape, ' source shape: ', source.shape)
#     num_blocks, num_heads, block_size, head_size = cache.shape
#     bsz, max_blocks_per_seq = block_tables.shape
#     needed_blocks = (lengths + block_size - 1) // block_size

#     if type == "prefill":
#         for i in range(bsz):
#             seq_len = lengths[i]
#             block_num = needed_blocks[i]
#             token_id = 0
#             for block_idx in range(block_num - 1):
#                 print('Source of block size shape and cache of block size: ', source[i][:, token_id : token_id + block_size, :].shape, cache[block_tables[i][block_idx]].shape)
#                 cache[block_tables[i][block_idx]] = source[i][:, token_id : token_id + block_size, :] #source[i][token_id : token_id + block_size].permute(1, 0, 2)
#                 token_id += block_size
#             cache[block_tables[i][block_num - 1], :, : seq_len - token_id, :] = source[i][:, token_id : seq_len, :]
            
#             # source[i][token_id:seq_len].permute(
#             #     1, 0, 2
#             # )
#     elif type == "decoding":
#         assert source.size(2) == 1, "seq_len should be equal to 1 when decoding."
#         source = source.squeeze(2)
#         slot_idx = (lengths + block_size - 1) % block_size
#         for i in range(bsz):
#             cache[block_tables[i, needed_blocks[i] - 1], :, slot_idx[i], :] = source[i]

#     return cache

# def convert_kvcache(cache, lengths, block_tables, pad_id=0):
#     """
#     Func: convert key/value cache for calculation

#     Args:   cache: shape [num_blocks, num_heads, block_size, head_size]
#             lengths: key/value length
#             block_tables
#             pad_id: padded_id
#     """
#     num_blocks, num_heads, block_size, head_size = cache.shape

#     needed_blocks = (lengths + block_size - 1) // block_size
#     num_remaing_tokens = lengths % block_size
#     num_remaing_tokens[num_remaing_tokens == 0] += block_size
#     bsz = block_tables.shape[0]
#     seq_len = max(lengths)
#     padded_cache = []
#     for i in range(bsz):
#         # print('Concat caches shapes: ', cache[block_tables[i][: needed_blocks[i] - 1]].shape, cache[block_tables[i][needed_blocks[i] - 1], :, : num_remaing_tokens[i], :].shape)
#         _cache = torch.cat(
#             (
#                 cache[block_tables[i][: needed_blocks[i] - 1]].permute((0, 2, 1, 3)).reshape(-1, num_heads, head_size), #.reshape(num_heads, -1, head_size), #
#                 cache[block_tables[i][needed_blocks[i] - 1], :, : num_remaing_tokens[i], :].permute(1, 0, 2),
#             ),
#             dim=0,
#         )
#         padding = seq_len - _cache.size(0)
#         if padding > 0:
#             _cache = F.pad(_cache, (0, 0, 0, 0, padding, 0), value=pad_id)
#             # print('_cache shape: ', _cache.shape)
#         padded_cache.append(_cache)
#     return torch.stack(padded_cache, dim=0).permute(0, 2, 1, 3)

def copy_to_cache(source, cache, lengths, block_tables, type: str = "prefill"):
    """
    Func: copy key/value into key/value cache.

    Args:   key/value(source): shape [bsz,seq_len,num_heads,head_size]
            cache: shape [num_blocks, num_kv_heads, head_size, block_size]
            lengths: key/value lengths
            block_tables
    """
    num_blocks, num_heads, block_size, head_size = cache.shape
    bsz, max_blocks_per_seq = block_tables.shape
    needed_blocks = (lengths + block_size - 1) // block_size

    if type == "prefill":
        for i in range(bsz):
            seq_len = lengths[i]
            block_num = needed_blocks[i]
            token_id = 0
            for block_idx in range(block_num - 1):
                cache[block_tables[i][block_idx]] = source[i][token_id : token_id + block_size].permute(1, 0, 2)
                token_id += block_size
            cache[block_tables[i][block_num - 1], :, : seq_len - token_id, :] = source[i][token_id:seq_len].permute(
                1, 0, 2
            )
    elif type == "decoding":
        assert source.size(1) == 1, "seq_len should be equal to 1 when decoding."
        source = source.squeeze(1)
        slot_idx = (lengths + block_size - 1) % block_size
        for i in range(bsz):
            cache[block_tables[i, needed_blocks[i] - 1], :, slot_idx[i], :] = source[i]

    return cache


def convert_kvcache(cache, lengths, block_tables, pad_id=0):
    """
    Func: convert key/value cache for calculation

    Args:   cache: shape [num_blocks, num_heads, block_size, head_size]
            lengths: key/value length
            block_tables
            pad_id: padded_id
    """
    num_blocks, num_heads, block_size, head_size = cache.shape

    needed_blocks = (lengths + block_size - 1) // block_size
    num_remaing_tokens = lengths % block_size
    num_remaing_tokens[num_remaing_tokens == 0] += block_size
    bsz = block_tables.shape[0]
    seq_len = max(lengths)
    padded_cache = []
    for i in range(bsz):
        _cache = torch.cat(
            (
                cache[block_tables[i][: needed_blocks[i] - 1]].permute((0, 2, 1, 3)).reshape(-1, num_heads, head_size),
                cache[block_tables[i][needed_blocks[i] - 1], :, : num_remaing_tokens[i], :].permute(1, 0, 2),
            ),
            dim=0,
        )
        padding = seq_len - _cache.size(0)
        if padding > 0:
            _cache = F.pad(_cache, (0, 0, 0, 0, padding, 0), value=pad_id)
        padded_cache.append(_cache)
    return torch.stack(padded_cache, dim=0)

class PagedAttention:

    @staticmethod
    def prefill_forward(
        idx,
        q: torch.Tensor,  # [batch_size, seq_len, num_heads, head_size]
        k: torch.Tensor,  # [batch_size, seq_len, num_kv_heads, head_size]
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_size]
        v_cache: torch.Tensor,
        context_lengths: torch.Tensor,  # [num_seqs]
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    ):
        # print('k_cache shape: ', k_cache.shape)
        # Firt, do shape verification
        bsz, seq_len, num_heads, head_size = q.shape
        num_kv_heads = k.shape[-2]
        assert num_heads % num_kv_heads == 0, "num_kv_heads should be divisible by num_heads"
        num_kv_groups = num_heads // num_kv_heads
        block_size = k_cache.size(-2)
        assert q.shape[0] == k.shape[0] == v.shape[0] == block_tables.shape[0]
        block_tables.shape[-1] * block_size

        # Copy kv to memory(rotary embedded)
        # if idx == 0:
        #     print('Caching in prefill: ', k[:3], k.shape)

        copy_to_cache(k.permute(0,2,1,3), k_cache.view(v_cache.shape), lengths=context_lengths, block_tables=block_tables)
        copy_to_cache(v.permute(0,2,1,3), v_cache, lengths=context_lengths, block_tables=block_tables)

    @staticmethod
    def decode_forward(
        idx,
        q: torch.Tensor,  # [bsz, 1, num_heads, head_size]
        k: torch.Tensor,  # [bsz, 1, num_kv_heads, head_size]
        v: torch.Tensor,
        k_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_size]
        v_cache: torch.Tensor,
        context_lengths: torch.Tensor,  # [num_seqs]: input_lengths + output_lengths
        block_tables: torch.Tensor,  # [num_seqs,max_blocks_per_sequence]
    ):
        bsz, q_length, num_heads, head_size = q.shape

        num_kv_heads = k.shape[-2]
        assert num_heads % num_kv_heads == 0, "num_kv_heads should be divisible by num_heads"
        num_kv_groups = num_heads // num_kv_heads

        assert q.shape[0] == k.shape[0] == v.shape[0] == block_tables.shape[0]

        copy_to_cache(k.permute(0,2,1,3), k_cache.view(v_cache.shape), lengths=context_lengths, block_tables=block_tables, type="decoding")
        copy_to_cache(v.permute(0,2,1,3), v_cache, lengths=context_lengths, block_tables=block_tables, type="decoding")

        k = convert_kvcache(k_cache.view(v_cache.shape), context_lengths, block_tables).permute(0,2,1,3)  # bsz, seqlen,
        v = convert_kvcache(v_cache, context_lengths, block_tables).permute(0,2,1,3)
        # print('Post convert: ', k.shape, v.shape)
        # if idx == 0:
        #     print('Decode k cache: ', k[:1], k.shape)
        return k,v #k.permute(0,2,1,3),v.permute(0,2,1,3)