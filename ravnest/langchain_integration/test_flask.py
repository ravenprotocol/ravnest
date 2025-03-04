import requests
import json

def test_completion(api_url, api_key, prompt, max_tokens=100, stream=False):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    response = requests.post(f"{api_url}/v1/completions", headers=headers, json=payload, stream=stream)
    
    if stream:
        for line in response.iter_lines():
            if line:
                print(json.loads(line))
    else:
        print(response.json())

if __name__ == "__main__":
    API_URL = "http://localhost:8080"  
    API_KEY = "admin_secret_api_key" 
    PROMPT = "Tell me a joke about AI."
    
    print("Testing non-streaming response:")
    test_completion(API_URL, API_KEY, PROMPT, max_tokens=50, stream=False)
    
    print("\nTesting streaming response:")
    test_completion(API_URL, API_KEY, PROMPT, max_tokens=50, stream=True)