#Code based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L151

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs 
from transformers.processing_utils import Unpack
from typing import List, Optional, Union, Tuple
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging
from .paged_attention_layer import PagedAttention
from ...strings import *

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LlamaPagedAttentionLayer():

    @staticmethod
    def forward(
                idx,
                hidden_states: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor] = None,
                native_attention_layer = None,
                prefill = True,
                k_cache = None,
                v_cache = None,
                block_tables=None,
                context_lengths=None,
                **kwargs: Unpack[FlashAttentionKwargs],
            ):

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, native_attention_layer.head_dim)

            query_states = native_attention_layer.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = native_attention_layer.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = native_attention_layer.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # print('Key states: ', key_states[-1,-1,-1])
            if idx == 0:
                print('\n Key states: ', key_states.shape, key_states[0,-1,:10])
            if prefill:
                PagedAttention.prefill_forward(
                    idx,
                    query_states, 
                    key_states,
                    value_states,
                    k_cache,
                    v_cache,
                    context_lengths,
                    block_tables
                )
            else:
                key_states, value_states = PagedAttention.decode_forward(
                    idx,
                    query_states,
                    key_states,
                    value_states,
                    k_cache,
                    v_cache,
                    context_lengths,
                    block_tables
                )
            
            attention_interface: Callable = eager_attention_forward
            if native_attention_layer.config._attn_implementation != "eager":
                if native_attention_layer.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[native_attention_layer.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                native_attention_layer,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not native_attention_layer.training else native_attention_layer.attention_dropout,
                scaling=native_attention_layer.scaling,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = native_attention_layer.o_proj(attn_output)
            return attn_output, attn_weights


class LlamaPagedAttentionDecoderLayer():

    @staticmethod
    def paged_attention_forward(
                idx,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: Optional[bool] = False,
                use_cache: Optional[bool] = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
                native_decoder_layer=None, 
                prefill = True,
                k_cache = None,
                v_cache = None,
                block_tables=None,
                context_lengths=None,
                **kwargs: Unpack[FlashAttentionKwargs],
            ):

        residual = hidden_states

        hidden_states = native_decoder_layer.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = LlamaPagedAttentionLayer.forward(
            idx,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            native_attention_layer = native_decoder_layer.self_attn,
            prefill = prefill,
            k_cache = k_cache,
            v_cache = v_cache,
            block_tables=block_tables,
            context_lengths=context_lengths,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # else:


        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = native_decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = native_decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
        

class BaseLlamaModel():
    def __init__(self, model, stage_idx, node_type, start_index, end_index):
        self.config = model.config
        self.model = model

        if self.model.__class__.__name__ == "LlamaModel":
            self.llama_model = model
        else:
            self.llama_model = model.model
        
        self.stage_idx = stage_idx
        self.node_type = node_type
        self.start_index = start_index
        self.end_index = end_index
        
        
    def forward(self, 
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids:Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            hidden_states: Optional[torch.FloatTensor] = None,
            prefill=True,
            k_caches = None,
            v_caches = None,
            block_tables=None,
            context_lengths=None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],            
            ):
        
        logger = logging.get_logger(__name__)
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.node_type == NodeTypes.ROOT:
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape[:2]
            elif inputs_embeds is not None:
                batch_size, seq_length = inputs_embeds.shape[:2]
            if inputs_embeds is None:
                inputs_embeds = self.llama_model.embed_tokens(input_ids)
            
            hidden_states = inputs_embeds
            device = hidden_states.device
        
        else:
            input_shape = hidden_states.shape[:-1]
            batch_size, seq_length = input_shape
            device = hidden_states.device
            

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=device
            )


        if output_attentions:
            logger.warning_once("output_attentions=True is not supported for pipeline models at the moment.")
            output_attentions = False
        if output_hidden_states:
            logger.warning_once("output_hidden_states=True is not supported for pipeline models at the moment.")
            output_hidden_states = False
        if use_cache:
            logger.warning_once("use_cache=True is not supported for pipeline models at the moment.")
            use_cache = False

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.llama_model._update_causal_mask(
            attention_mask, hidden_states, cache_position, past_key_values, output_attentions
        )

        position_embeddings = self.llama_model.rotary_emb(hidden_states, position_ids)
        print('\n Position embeddings: ', position_embeddings[0].shape)
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.llama_model.layers[self.start_index:self.end_index]):
        
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # layer_outputs = decoder_layer(
            #     hidden_states,
            #     attention_mask=causal_mask,
            #     position_ids=position_ids,
            #     past_key_value=past_key_values,
            #     output_attentions=output_attentions,
            #     use_cache=use_cache,
            #     cache_position=cache_position,
            #     position_embeddings=position_embeddings,
            #     **flash_attn_kwargs,
            # )
            layer_outputs = LlamaPagedAttentionDecoderLayer.paged_attention_forward(
                i,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                native_decoder_layer=decoder_layer,
                prefill=prefill,
                k_cache=k_caches[i],
                v_cache=v_caches[i],
                block_tables=block_tables,
                context_lengths=context_lengths,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.node_type == NodeTypes.LEAF:
            hidden_states = self.llama_model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.node_type == NodeTypes.LEAF:
            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
            return output if return_dict else output.to_tuple()
        
        return {"hidden_states":hidden_states}

class LlamaForCausalLMPipelineStage(BaseLlamaModel):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        prefill=True,
        k_caches = None,
        v_caches = None,
        block_tables=None,
        context_lengths=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            hidden_states=hidden_states,
            prefill=prefill,
            k_caches = k_caches,
            v_caches = v_caches,
            block_tables=block_tables,
            context_lengths=context_lengths,
            **kwargs,
        )
        
        if self.node_type == NodeTypes.LEAF:
            hidden_states = outputs[0]

            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.model.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                loss = self.model.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            hidden_states = outputs.get("hidden_states")
            return {"hidden_states": hidden_states}