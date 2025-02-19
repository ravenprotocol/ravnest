import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ravnest.lazy_init.lazy_context import LazyInitContext
from ravnest import Node

device = torch.device('cuda')

# model_name = 'meta-llama/Llama-3.2-1B' 
# model_name = 'meta-llama/Llama-3.1-8B'
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'

tokenizer = AutoTokenizer.from_pretrained(model_name)#, legacy=False)


init_ctx = LazyInitContext()
with init_ctx:
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    torch_dtype=torch.float16,
                                                    device_map="cuda")


model.eval()

print('Model: ', model)
prompt = "Black holes are"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print('inputs: ', inputs)

node = Node(model=model,
            device=device,
            dtype='float16',
            batch_size=1,
            mode='inference',
            seq_length=inputs['input_ids'].shape[1],
            backend='nccl',
            cluster_length=3,
            reduce_factor=1)

node.model.eval()

output = node.generate(input_ids=inputs['input_ids'], max_seq_length=50, tokenizer=tokenizer) #node.no_grad_forward(**inputs)

print('Output: ', output)

# torchrun --nnodes=3 --nproc_per_node=1 --node_rank=0 --master_addr=localhost --master_port=29500 examples/llama/inference.py
