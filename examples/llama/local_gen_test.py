import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ravnest.pipeline_split.modeling import get_pipeline_stage_submod

device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B', legacy=False)
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B',
                                             torch_dtype=torch.float16,
                                             device_map="cpu")
model.eval()

# print('State dict: ', model.state_dict())
# print('Lm head weights: ', model.lm_head.weight)

# stage_0_submod = get_pipeline_stage_submod(model=model, stage_idx=0, node_type='root', start_index=0, end_index=11)
# stage_1_submod = get_pipeline_stage_submod(model=model, stage_idx=1, node_type='stem', start_index=11, end_index=22)
# stage_2_submod = get_pipeline_stage_submod(model=model, stage_idx=2, node_type='leaf', start_index=22, end_index=32)

def generate(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")#.to(device)
    input_ids = inputs["input_ids"]

    for i in range(max_length):
        print(f'{i} - processing...')
        with torch.no_grad():
            stage_0_submod = get_pipeline_stage_submod(model=model, stage_idx=0, node_type='root', start_index=0, end_index=11)#.cuda()
            outputs_stage_0 = stage_0_submod.forward(**inputs)
            print('Stage 0 op: ', outputs_stage_0)
            del stage_0_submod

            stage_1_submod = get_pipeline_stage_submod(model=model, stage_idx=1, node_type='stem', start_index=11, end_index=22)#.cuda()
            outputs_stage_1 = stage_1_submod.forward(hidden_states=outputs_stage_0['hidden_states'])  
            print('Stage 1 op: ', outputs_stage_1)
            del stage_1_submod

            stage_2_submod = get_pipeline_stage_submod(model=model, stage_idx=2, node_type='leaf', start_index=22, end_index=32)#.cuda()
            outputs_stage_2 = stage_2_submod.forward(hidden_states=outputs_stage_1['hidden_states'])
            print('Stage 2 op: ', outputs_stage_2)
            del stage_2_submod
            
            logits = outputs_stage_2.logits[:, -1, :]
            print('logits: ', logits)
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            inputs = {"input_ids": input_ids}

    return tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

prompt = "Once upon a time"#"The moon is shining"
generated_text = generate(prompt, max_length=10)
print(generated_text)
