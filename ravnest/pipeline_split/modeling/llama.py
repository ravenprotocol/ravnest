from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs 
from transformers.processing_utils import Unpack
import torch
from typing import List, Optional, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.utils import logging
from ...strings import *

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
        ) #inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.llama_model.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        for decoder_layer in self.llama_model.layers[self.start_index:self.end_index]:
        
            if output_hidden_states:
                all_hidden_states += (hidden_states,)


            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
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