from typing import Set
from pathlib import Path
from functools import reduce
from typing import Mapping, List, OrderedDict, Optional
import os
import torch
import torch.nn as nn
import json

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = "_extra_state"

# Taken from: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/shard/utils.py
def set_tensors_to_none(model: nn.Module, exclude: Set[nn.Module] = set()) -> None:
    """Set all parameters and buffers of model to None

    Args:
        model (nn.Module): The model to set
    """
    if model in exclude:
        return
    for child in model.children():
        set_tensors_to_none(child, exclude=exclude)
    for n, p in model.named_parameters(recurse=False):
        setattr(model, n, None)
    for n, buf in model.named_buffers(recurse=False):
        setattr(model, n, None)

def load_shard_state_dict(checkpoint_file: Path, use_safetensors: bool = False):
    """
    load shard state dict into model
    """
    if use_safetensors and not checkpoint_file.suffix == ".safetensors":
        raise Exception("load the model using `safetensors`, but no file endwith .safetensors")
    if use_safetensors:
        from safetensors.torch import load_file as safe_load_file

        return safe_load_file(checkpoint_file)
    else:
        return torch.load(checkpoint_file, map_location=torch.device("cpu"))

def load_index_file(json_path:str):
    with open(json_path, "r") as f:
        index = json.load(f)

    metadata = index.get("metadata", None)
    weight_map = index.get("weight_map", None)
    root_path = Path(json_path).absolute().parent
    return metadata, weight_map, root_path

# Based on get_non_persistent_buffers_set from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/utils/common.py#L82
def get_non_persistent_buffers_set(
    module, memo: Optional[Set[nn.Module]] = None, prefix: str = "", remove_duplicate: bool = True
):
    r"""
    Args:
        memo: a memo to store the set of modules already added to the result
        prefix: a prefix that will be added to the name of the module
        remove_duplicate: whether to remove the duplicated module instances in the result
            or not
    """

    if memo is None:
        memo = set()
    self_non_persistent_set = set()
    if module not in memo:
        if remove_duplicate:
            memo.add(module)
        self_non_persistent_set = set(
            map(lambda key: prefix + ("." if prefix else "") + key, module._non_persistent_buffers_set)
        )
        for name, sub_module in module._modules.items():
            if sub_module is None:
                continue
            submodule_prefix = prefix + ("." if prefix else "") + name
            child_non_persistent_set = get_non_persistent_buffers_set(
                sub_module, memo, submodule_prefix, remove_duplicate
            )
            self_non_persistent_set = set.union(self_non_persistent_set, child_non_persistent_set)
    return self_non_persistent_set

# Based on load_state_dict_into_model from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/checkpoint_io/utils.py#L670
def load_state_dict_into_model(
    model: nn.Module, state_dict: torch.Tensor, missing_keys: List, strict: bool = False, load_sub_module: bool = True
):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

    unexpected_keys: List[str] = []
    sub_missing_keys: List[str] = []
    error_msgs: List[str] = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = OrderedDict(state_dict)
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module: nn.Module, state_dict, prefix="", load_sub_module: bool = True):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, sub_missing_keys, unexpected_keys, error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if strict or len([key for key in state_dict if key.startswith(prefix)]) > 0:
            module._load_from_state_dict(*args)
        if load_sub_module:
            for name, child in module._modules.items():
                if child is not None:
                    load(child, state_dict, prefix + name + ".")

    load(model, state_dict, "", load_sub_module)
    del load

    missing_keys = missing_keys.append(sub_missing_keys)

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs = [
                "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in unexpected_keys))
            ]
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
            )

# Based on load_sharded_model from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/checkpoint_io/hybrid_parallel_checkpoint_io.py
def load_stage_weights_from_checkpoint(model, checkpoint_index_file:Path):
    use_safetensors = False
    if "safetensors" in checkpoint_index_file.name:
        use_safetensors = True

    _, weight_map, root_path = load_index_file(checkpoint_index_file)

    loaded_file = set()

    missing_keys = []
    missing_file_keys = []

    def _load(name: str):
        if name not in weight_map:
            missing_file_keys.append(name)
            return
        filename = weight_map[name]

        # If this param/buffer has been loaded before, directly return.
        if filename in loaded_file:
            return

        file_path = os.path.join(root_path, filename)
        state_dict = load_shard_state_dict(Path(file_path), use_safetensors)
        load_state_dict_into_model(
            model, state_dict, missing_keys=missing_keys, strict=False, load_sub_module=True
        )
        loaded_file.add(filename)

    # Load parameters.
    for name, _ in model.named_parameters():
        _load(name)

    # Load buffers.
    non_persistent_buffers = get_non_persistent_buffers_set(model)
    for name, buf in model.named_buffers():
        if buf is not None and name not in non_persistent_buffers:
            _load(name)

    # Load extra states.
    extra_state_key = _EXTRA_STATE_KEY_SUFFIX
    if (
        getattr(model.__class__, "get_extra_state", nn.Module.get_extra_state)
        is not nn.Module.get_extra_state
    ):
        _load(extra_state_key)

    # Update master params if mixed-precision training is enabled.
    # model_before_wrapping.update_master_params()

    print(f"The model has been successfully loaded from sharded checkpoint: {root_path}.")

    if len(missing_keys) == 0:
        raise RuntimeError(
            "No weigth is loaded into the model. Please check the checkpoint files and the model structure."
        )

    remain_keys = reduce(lambda a, b: a & b, map(set, missing_keys))
    remain_keys = remain_keys.union(set(missing_file_keys))
    if len(remain_keys) > 0:
        print(f"The following keys are not loaded from checkpoint: {remain_keys}")

    