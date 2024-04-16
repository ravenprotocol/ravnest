import os
import json
import torch
import asyncio
import nvidia_smi
import numpy as np
import _pickle as cPickle
from contextlib import contextmanager
from typing import TypeVar, AsyncIterable, Optional, AsyncIterator
from .protos.tensor_pb2 import TensorChunk, SendTensor
from .protos.server_pb2 import ReduceChunk, DataChunk

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
        
def generate_data_stream(data, ring_id=None, type=None):
    if not isinstance(data, bytes):
        obj = cPickle.dumps(data)
    else:
        obj = data
    
    file_size = len(obj)
    blocksize = 2*1024*1024
    if file_size > blocksize:
        for i in range(0, file_size, blocksize):
            data = obj[i:i+blocksize]
            data_chunk = DataChunk(buffer=data, data_size=file_size)
            yield ReduceChunk(ring_id=ring_id, data_chunk=data_chunk)
        
    else:
        data_chunk = DataChunk(buffer=obj, data_size=file_size)
        yield ReduceChunk(ring_id=ring_id, data_chunk=data_chunk)

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
        optimizer_param.grad = model_param.grad #.clone()

@torch.no_grad()
def load_optim_weights_into_model(model, optimizer):
    for model_param, optimizer_param in zip(model.parameters(), optimizer_params(optimizer)):
        model_param.data = optimizer_param.data

@torch.no_grad()
def load_model_weights_into_optim(model, optimizer):
    for model_param, optimizer_param in zip(model.parameters(), optimizer_params(optimizer)):
        optimizer_param.data = model_param.data

@torch.no_grad()
def get_trainable_param_names(model):
    param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_names.append(name)
    return param_names

@torch.no_grad()
def load_state_dict_conserve_versions(model, state_dict):
    # model_state_dict = model.state_dict(keep_vars=True)
    # for k, v in state_dict.items():
    #     model_state_dict[k].data = v.data
    for name, param in model.named_parameters():
        param.data = state_dict[name].data

def load_node_json_configs(node_name=None):
    with open('node_data/nodes/{}.json'.format(node_name)) as f:
        data = f.read()
    parsed_json = json.loads(data)
    
    submod_file = None
    dir_files = os.listdir(parsed_json['template_path'])
    for file in dir_files:
        if 'submod_' in file:
            file_name_parts = file.split('_')
            submod_file = '_'.join(file_name_parts[:2])
    
    parsed_json['submod_file'] = submod_file
    parsed_json['param_addresses'] = parsed_json['param_addresses'][0]
    parsed_json['ring_ids'] = {int(key): value for key, value in parsed_json['ring_ids'].items()}
    
    return parsed_json

def create_chunks(data, size):
    chunked_data = {}
    for key, val in data.items():
        split_axis = np.argmax(val.shape)
        chunked_data[key] = {}
        chunked_data[key]['data'] = list(torch.chunk(val, chunks=size, dim=split_axis))
        chunked_data[key]['split_axis'] = split_axis

    return chunked_data

def check_gpu_usage():
    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, round(info.total * 1e-9, 2), round(info.free * 1e-9, 2), round(info.used * 1e-9, 2)))
            
    nvidia_smi.nvmlShutdown()