"""
OpenAI-compatible chat completions API server for Ravnest distributed inference.

Non-streaming, single-request-at-a-time. Wraps InferenceEngine.generate().
"""

import time
import threading
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ravnest"
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 1


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


def create_app(engine, tokenizer):
    app = FastAPI(title="Ravnest Inference API")
    lock = threading.Lock()
    MAX_SEQ_LENGTH = 3000

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        # Validate messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages array is required and must not be empty")

        # Build prompt from messages
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        # Estimate prompt length and validate
        prompt_token_count = len(tokenizer.encode(prompt))
        total_seq_length = prompt_token_count + request.max_tokens

        if total_seq_length > MAX_SEQ_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"prompt ({prompt_token_count} tokens) + max_tokens ({request.max_tokens}) "
                       f"exceeds max sequence length ({MAX_SEQ_LENGTH})"
            )

        # Serialize requests (no concurrent inference)
        acquired = lock.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail="Server busy, try again later")

        try:
            start_time = time.time()

            # max_seq_lengths = max NEW tokens per prompt (parallel array)
            outputs = engine.generate(
                prompt_list=[prompt],
                max_seq_lengths=[request.max_tokens],
                top_k=request.top_k,
                temperature=request.temperature,
            )

            elapsed = time.time() - start_time

            # outputs is a list of generated strings (full sequence including prompt)
            generated_text = outputs[0] if outputs else ""

            # Strip prompt from output if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            generated_text = generated_text.strip()

            completion_tokens = len(tokenizer.encode(generated_text))

            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=generated_text),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_token_count,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_token_count + completion_tokens,
                ),
            )

            print(f"[api] Request completed in {elapsed:.2f}s, "
                  f"prompt={prompt_token_count} tokens, completion={completion_tokens} tokens")

            return response

        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        finally:
            lock.release()

    return app
