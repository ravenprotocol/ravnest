import requests
from langchain_core.language_models import LLM
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from operator import itemgetter

class RavnestLLM(LLM, BaseModel):
    api_url: str = Field(..., description="Flask API endpoint for Ravnest model")
    api_key: str = Field(..., description="API Key for authentication")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    streaming: bool = Field(False, description="Enable or disable streaming responses")

    @property
    def _llm_type(self) -> str:
        return "ravnest_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "stream": self.streaming
        }
        
        response = requests.post(self.api_url, json=payload, headers=headers)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

rendered_tools = render_text_description([multiply])

system_prompt = f"""
You are an assistant with access to tools. Here is a tool you can use:

{rendered_tools}

Given the user input, return a JSON object with:
- `name`: The tool name (as a string)
- `arguments`: A dictionary containing the required arguments with values.

"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

model = RavnestLLM(
    api_url="http://localhost:8080/v1/completions",
    api_key="admin_secret_api_key"
)

# chain = prompt | model | JsonOutputParser()
# chain_response = chain.invoke({"input": "what's thirteen times 4"})

# print('Chain Response: ', chain_response)

chain = prompt | model | JsonOutputParser() | itemgetter("arguments") | multiply
final_response = chain.invoke({"input": "what's thirteen times 4"})

print('Final response: ', final_response)