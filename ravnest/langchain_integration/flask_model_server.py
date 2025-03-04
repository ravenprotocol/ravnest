from flask import Flask, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

app = Flask(__name__)
API_KEYS = {"admin_secret_api_key", "test_secret_api_key"}  

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                             torch_dtype=torch.float16, 
                                             device_map="auto")
model.eval()

def authenticate():
    api_key = request.headers.get("Authorization")
    if not api_key or api_key.replace("Bearer ", "") not in API_KEYS:
        return jsonify({"error": "Unauthorized"}), 401

def format_prompt(user_input):
    return f"""### Instruction:\nYou are a helpful AI assistant. Answer the user's request concisely.\n### User:\n{user_input}\n### AI assistant:\n"""

def generate_response(prompts, max_tokens, stream):
    formatted_prompts = [format_prompt(p) for p in prompts]
    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, eos_token_id=tokenizer.eos_token_id)
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    responses = []
    
    for text in generated_texts:
        match = re.search(r'### AI assistant:\s*(.*)', text, re.DOTALL)
        cleaned_response = match.group(1).strip() if match else text.strip()
        responses.append(cleaned_response.split("###")[0])
    
    if stream:
        def stream_responses():
            for response in responses:
                yield json.dumps({"choices": [{"text": response}]}) + "\n"
        return Response(stream_responses(), content_type='application/json')
    else:
        return jsonify({"choices": [{"text": r} for r in responses]})

@app.route('/v1/completions', methods=['POST'])
def completion():
    auth_response = authenticate()
    if auth_response:
        return auth_response
    
    data = request.json
    prompts = [data.get('prompt', "")] if isinstance(data.get('prompt'), str) else data.get('prompt', [])
    max_tokens = data.get('max_tokens', 1024)
    stream = data.get('stream', False)
    
    if not prompts:
        return jsonify({"error": "No prompt provided"}), 400
    
    return generate_response(prompts, max_tokens, stream)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
