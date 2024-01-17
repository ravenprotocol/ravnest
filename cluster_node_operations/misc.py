import torch

@torch.no_grad()
def get_trainable_parameters(model):
    # print("Trainable Parameters:")
    data_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            data_dict[name] = param
    return data_dict