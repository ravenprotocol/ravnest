import torch
import math
from .memory_tracker import MemoryTracker
from ..strings import *
from ..utils import *

MAX_NUM_TOKENS = 1500

class InferenceEngine():

    def __init__(self, node, tokenizer, track_mem_usage=True):
        self.node = node
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.node_type = self.node.node_type
        self.comm_session = self.node.comm_session
        self.is_pipelining = False
        self.track_mem_usage = track_mem_usage
        if self.track_mem_usage:
            self.memory_tracker = MemoryTracker(self.comm_session, self.node.device)
            self.memory_tracker.update_metrics()

    def get_microbatch_inputs(self, start_id, end_id, input_ids=None, **kwargs):
        microbatch_kwargs = {}
        for k, v in kwargs.items():
            microbatch_kwargs[k] = v[start_id:end_id]
        return input_ids[start_id:end_id], microbatch_kwargs

    def tokenize_and_pad_batch(self, prompts: list, add_special_tokens: bool = True):
        prompt_tokens = self.tokenizer(prompts, add_special_tokens=add_special_tokens, padding=True, padding_side='left', return_tensors="pt")
        return prompt_tokens

    def tokenizer_decode_batch(self, batch_input_ids):
        decoded_outputs = []
        batch_input_ids = batch_input_ids.cpu()
        for sequence in batch_input_ids:
            decoded_outputs.append(self.tokenizer.decode(sequence, skip_special_tokens=True))
        return decoded_outputs
    
    def configure_pipelining(self, input_ids=None, max_seq_lengths=None):

        if self.node_type == NodeTypes.ROOT:
            self.comm_session.broadcast_metadata_objects(max_seq_lengths)
        else:
            max_seq_lengths = [0] * input_ids.shape[0]
            self.comm_session.broadcast_metadata_objects(max_seq_lengths)
        
        max_seq_length_in_batch = max(max_seq_lengths)
        bs,seq_length = input_ids.shape[:2]
        self.comm_session.feedback_shape[0] = bs
        mbs_list = None
        if bs * max_seq_length_in_batch > MAX_NUM_TOKENS:
            self.is_pipelining = True
            mbs = max(1, MAX_NUM_TOKENS // max_seq_length_in_batch)
            num_microbatches = math.ceil(bs / mbs)
            mbs_list = [mbs for i in range(num_microbatches - 1)]
            mbs_list.append(bs - mbs * (num_microbatches - 1))

        return bs, mbs_list, seq_length, max_seq_length_in_batch, max_seq_lengths

    def pipelined_forward(self, bs, mbs_list, input_ids=None, **kwargs):
        if self.node_type == NodeTypes.LEAF:
            batch_output_logits = torch.empty(
                (bs, self.comm_session.forward_input_shapes[0][1], self.node.model.config.vocab_size),#self.comm_session.forward_input_shapes[0][2]),
                dtype=self.comm_session.dtype,
                device=torch.cuda.current_device(),
            )
        for mb_id, mbs in enumerate(mbs_list):
            # if self.node_type = NodeTypes.ROOT:
            if mb_id == len(mbs_list) - 1:
                start = mb_id * mbs_list[0]
            else:
                start = mb_id * mbs
            end = start + mbs #min(start + mbs, bs)
            microbatch_input_ids, microbatch_kwargs = self.get_microbatch_inputs(start, end, input_ids=input_ids, **kwargs)
            self.comm_session.forward_input_shapes[0][0] = mbs
            microbatch_outputs = self.node.no_grad_forward(input_ids=microbatch_input_ids, **microbatch_kwargs)
            # else:
                # self.comm_session.forward_input_shapes[0][0] = mbs
                # microbatch_outputs = self.no_grad_forward()

            if self.node_type == NodeTypes.LEAF:
                batch_output_logits[start:end, ...] = microbatch_outputs.logits

        if self.node_type == NodeTypes.LEAF:
            return batch_output_logits
        
        return None

    def batch_forward(self, bs=None, mbs_list=None, input_ids = None, **kwargs):
        # print('\nTorch memory allocated and reserved before batch: ', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        if self.is_pipelining:
            outputs = self.pipelined_forward(bs, mbs_list, input_ids=input_ids, **kwargs)
        else:
            self.comm_session.forward_input_shapes[0][0] = bs
            outputs = self.node.no_grad_forward(input_ids=input_ids, **kwargs)
            if outputs is not None:
                outputs = outputs.logits
        # print('Torch memory allocated and reserved after batch: ', torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
        if self.track_mem_usage:
            self.memory_tracker.update_metrics()
        return outputs

    def broadcast_prompt_list(self, prompt_list):
        if self.node_type == NodeTypes.ROOT:
            batch_size = len(prompt_list)
            self.comm_session.broadcast_metadata(torch.tensor(batch_size).cuda())
        else:
            batch_size = torch.tensor(0).cuda()
            self.comm_session.broadcast_metadata(batch_size)
            print('Recieved batch size: ', batch_size)
            batch_size = int(batch_size.item())
            prompt_list = ['' for _ in range(batch_size)]
        print('Broadcasting prompt list: ', prompt_list)
        self.comm_session.broadcast_metadata_objects(prompt_list)
        print('Done prompt list: ', prompt_list)
        return prompt_list

    def is_generation_complete(self, is_generation_done, new_token_ids, num_generated_tokens, max_seq_lengths):
        eos_generated = (new_token_ids == self.tokenizer.eos_token_id)
        max_seq_length_generated = (torch.tensor(max_seq_lengths) <= num_generated_tokens).to(self.node.device)
        is_generation_done = eos_generated | max_seq_length_generated | is_generation_done
        return is_generation_done

    # @torch.no_grad()
    @torch.inference_mode()
    def _generate(self, input_ids=None, max_seq_lengths=None, top_k=1, temperature=1.0, **kwargs):
        
        bs, mbs_list, seq_length, max_seq_length_in_batch, max_seq_lengths = self.configure_pipelining(input_ids, max_seq_lengths)
        num_generated_tokens = 0
        is_generation_done = torch.tensor([False]*bs).to(self.node.device)
        pad_token_tensor = torch.tensor([self.tokenizer.pad_token_id]*bs).to(self.node.device)
        while num_generated_tokens < max_seq_length_in_batch:
            self.comm_session.forward_input_shapes[0][1] = seq_length #input_ids.shape[1]

            output_logits = self.batch_forward(bs=bs, mbs_list=mbs_list, input_ids=input_ids, **kwargs)

            if self.node_type == NodeTypes.LEAF:
                last_token_logits = output_logits[:, -1, :]
                next_token_ids = sample_token_from_logits(last_token_logits, top_k, temperature)
                self.comm_session.trigger_feedback_send(next_token_ids)
            else:
                self.comm_session.start_feedback_recv()
                while not self.comm_session.feedback_recv_work_done():
                    time.sleep(0)
                next_token_ids = self.comm_session.feedback_ip
            
            seq_length += 1
            num_generated_tokens += 1
            next_token_ids = torch.where(is_generation_done, pad_token_tensor, next_token_ids)
            input_ids = torch.cat((input_ids, next_token_ids[:,None]), dim=-1)

            if kwargs.get('attention_mask', None) is not None:
                new_token_mask = kwargs['attention_mask'].new_ones((bs,1))
                kwargs['attention_mask'] = torch.cat((kwargs['attention_mask'], new_token_mask), axis=-1)

            is_generation_done = self.is_generation_complete(is_generation_done, next_token_ids, num_generated_tokens, max_seq_lengths)
            
            if torch.all(is_generation_done):
                break

        return input_ids #tokenizer_decode_batch(input_ids, self.tokenizer)

    def generate(self, prompt_list=None, max_seq_lengths=None, top_k=1, temperature=1.0):
        prompt_list = self.broadcast_prompt_list(prompt_list)        
        tokenized_and_padded_batch = self.tokenize_and_pad_batch(prompt_list).to(self.node.device)
        generated_tokens = self._generate(**tokenized_and_padded_batch, 
                                        max_seq_lengths=max_seq_lengths, 
                                        top_k=top_k, temperature=temperature)
        if self.track_mem_usage:
            self.memory_tracker.update_metrics()
        return self.tokenizer_decode_batch(generated_tokens)

