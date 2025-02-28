import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from ravnest.lazy_init.lazy_context import LazyInitContext
from ravnest import Node, InferenceEngine

device = torch.device('cuda')

# model_name = 'meta-llama/Llama-3.2-1B' 
# model_name = 'meta-llama/Llama-3.1-8B'
# model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
model_name = 'meta-llama/Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name)

# This lazy context mechanism is based on Colossal AI's lazy init: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/lazy/lazy_init.py
init_ctx = LazyInitContext()
with init_ctx:
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    torch_dtype=torch.float16,
                                                    device_map="cuda")


model.eval()

print('Model: ', model)

user_inputs = []
prompts = [f"""### Instruction:
You are a helpful AI assistant. Answer the user's request concisely.

### User:
{user_input}

### AI assistant:
""" for user_input in user_inputs]

max_seq_lengths = [50]*len(prompts)

node = Node(model=model,
            device=device,
            dtype='float16',
            batch_size=1,
            mode='inference',
            seq_length=5,
            backend='nccl',
            cluster_length=3,
            reduce_factor=1)

node.model.eval()

inference_engine = InferenceEngine(node, tokenizer)

outputs = inference_engine.generate(prompts, max_seq_lengths=max_seq_lengths)
print('Generated: ', outputs)

# torchrun --nnodes=3 --nproc_per_node=1 --node_rank=0 --master_addr=localhost --master_port=29500 examples/llama/inference_with_eng.py
