import torch
import asyncio
import _pickle as cPickle
from typing import TypeVar, AsyncIterable, Optional, AsyncIterator
from tensor_pb2 import TensorChunk, SendTensor

T = TypeVar("T")

async def aiter_with_timeout(iterable: AsyncIterable[T], timeout: Optional[float]) -> AsyncIterator[T]:
    """Iterate over an async iterable, raise TimeoutError if another portion of data does not arrive within timeout"""
    # based on https://stackoverflow.com/a/50245879
    iterator = iterable.aiter()
    while True:
        try:
            yield await asyncio.wait_for(iterator.anext(), timeout=timeout)
        except StopAsyncIteration:
            break


def generate_stream(data, type=None):
    if not isinstance(data, bytes):
        obj = cPickle.dumps(data)
    else:
        obj = data
    
    file_size = len(obj)
    blocksize = 2*1024*1024
    if file_size > blocksize:
        for i in range(0, file_size, blocksize):
            data = obj[i:i+blocksize]
            tensor_chunk = TensorChunk(buffer=data, type='$tensor', tensor_size=file_size)
            yield SendTensor(tensor_chunk=tensor_chunk, type=type)
        
    else:
        tensor_chunk = TensorChunk(buffer=obj, type='$tensor', tensor_size=file_size)
        yield SendTensor(tensor_chunk=tensor_chunk, type=type)
        

@torch.no_grad()
def current_model_params_clone(model):
    for param in model.parameters():
        yield param.clone()

@torch.no_grad()
def optimizer_params(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            yield param

@torch.no_grad()
def load_grads_into_optimizer(model, optimizer):
    for model_param, optimizer_param in zip(model.parameters(), optimizer_params(optimizer)):
        optimizer_param.grad = model_param.grad.clone()

@torch.no_grad()
def load_optim_weights_into_model(model, optimizer):
    for model_param, optimizer_param in zip(model.parameters(), optimizer_params(optimizer)):
        model_param.data = optimizer_param.data