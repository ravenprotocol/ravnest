import numpy as np
import torch
import os

def load_submodels(dir):
    num_submodels = len(os.listdir(dir))
    submodels = []
    for i in range(num_submodels):
        submodel_path = f'trained_submodels/submod_{i}.pt'
        submodel = torch.jit.load(submodel_path)
        submodels.append(submodel)
    return submodels

def forward(inp, submodels):
    for submodel in submodels:
        submodel.eval()
    with torch.no_grad():
        out = inp
        for submodel in submodels:
            out = submodel(out) if submodels.index(submodel) == 0 else submodel(*out)
    return out

def generate(x, submodels, steps=6):
    x = np.expand_dims(x, axis=0)
    x = torch.tensor(x)
    with torch.no_grad():
        for i in range(steps):
            out = forward(x, submodels)
            out_tensor = out[:, -1, :]
            out_token = torch.argmax(out_tensor, axis=-1, keepdim=True)
            x = torch.cat((x, out_token), axis=-1)

    print('Sorted Prediction: ', x[:, 6:])

submodels = load_submodels('trained_submodels')

test_input = np.array([2, 2, 1, 2, 0, 2])

generate(test_input, submodels, steps=6)
