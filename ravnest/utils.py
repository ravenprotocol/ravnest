import os
import glob
import json
import torch
import asyncio
import random
if torch.cuda.is_available():
    import nvidia_smi
import numpy as np
import _pickle as cPickle
from contextlib import contextmanager
from typing import TypeVar, AsyncIterable, Optional, AsyncIterator
from .protos.tensor_pb2 import TensorChunk, SendTensor
from .protos.server_pb2 import ReduceChunk, DataChunk, WeightsChunk

T = TypeVar("T")
FP16_MIN, FP16_MAX = torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
FP32_MIN, FP32_MAX = torch.finfo(torch.float32).min, torch.finfo(torch.float32).max

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

def generate_weights_stream(data):
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
            yield WeightsChunk(tensor_chunk=tensor_chunk)
        
    else:
        tensor_chunk = TensorChunk(buffer=obj, type='$tensor', tensor_size=file_size)
        yield WeightsChunk(tensor_chunk=tensor_chunk)
        
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

@torch.no_grad()
def load_optim_state(optimizer, state, model):
    id = 0
    new_state = {}
    for model_param_name, _ in model.named_parameters():
        new_state[id] = state[model_param_name]
        id += 1
    
    optim_state_dict = optimizer.state_dict()
    optim_state_dict['state'] = new_state
    optimizer.load_state_dict(optim_state_dict)

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
        chunked_data[key]['data'] = list(torch.tensor_split(val.to(torch.device('cpu')), size, split_axis))
        chunked_data[key]['split_axis'] = split_axis

    return chunked_data

def create_chunks_optim(data, size):
    chunked_optim_data = {}
    for key, val in data.items():
        optim_param_state = {}
        for k,v in val.items():
            optim_param_state[k] = {}
            if len(v.shape) < 1:
                v = v.reshape((1,))
                optim_param_state[k]['reshape'] = True
            else:
                optim_param_state[k]['reshape'] = False
            split_axis = np.argmax(v.shape)
            optim_param_state[k]['data'] = list(torch.tensor_split(v.to(torch.device('cpu')), size, split_axis))
            optim_param_state[k]['split_axis'] = split_axis
        chunked_optim_data[key] = optim_param_state
    return chunked_optim_data

def compress_tensor_float16(tensor):
    if tensor.dtype == torch.float64:
        tensor = tensor.clamp_(FP32_MIN, FP32_MAX).to(torch.float32)
    elif tensor.dtype == torch.float32:
        # tensor = tensor.to(torch.float32) #, copy=not allow_inplace)
        tensor = tensor.clamp_(FP16_MIN, FP16_MAX).to(torch.float16)
    return tensor

def extract_tensor_from_compression_float16(tensor, original_dtype):
    tensor = tensor.to(original_dtype)
    return tensor

def set_seed(seed=42):
    """Set the seed for random number generators across torch, numpy and random modules. Handles seed for torch.cuda based on GPU availability.
    
    :param seed: seed number, defaults to 42
    :type seed: int, optional
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed_all(42)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def check_gpu_usage():
    if torch.cuda.is_available():
        nvidia_smi.nvmlInit()

        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, round(info.total * 1e-9, 2), round(info.free * 1e-9, 2), round(info.used * 1e-9, 2)))
                
        nvidia_smi.nvmlShutdown()


def find_files_with_extension(root_folder, extension):
    file_paths = []
    for folder, _, files in os.walk(root_folder):
        matching_files = glob.glob(os.path.join(folder, f"submod*.{extension}"))
        file_paths.extend(matching_files)
    
    return file_paths

def model_fusion(cluster_id = 0):
    """Fuses state dictionaries from TorchScript submodels (.pt files) hosted on all Providers belonging to a cluster. The combined state dictionary is then saved at 'trained/trained_state_dict.pt'. This final state_dict can then be loaded into the main model using ``model.load_state_dict()`` for obtaining the final trained main model. 
    
    Make sure to set the ``save`` parameter of ``Trainer()`` instance to ``True`` for saving submodels post-training. Only then this method to work.

    :param cluster_id: ID of the cluster whose submodels need to be combined into the main model, defaults to 0
    :type cluster_id: int, optional
    """
    folder_path = 'node_data/cluster_{}'.format(cluster_id)
    if os.path.exists(folder_path):
        pt_files = find_files_with_extension(root_folder=folder_path, extension='pt')
        if len(pt_files) > 0:
            combined_state_dict = {}
            for file in pt_files:
                submod = torch.jit.load(file)
                submod_state_dict = {key.replace('L__self___', ''): value for key, value in submod.state_dict().items()}
                combined_state_dict.update(submod_state_dict)
            os.makedirs('trained', exist_ok=True)
            save_path = 'trained/trained_state_dict.pt'
            torch.save(combined_state_dict, save_path)
        else:
            print('{} path has no submodels'.format(folder_path))
    else:
        print('Submodels not found!')
