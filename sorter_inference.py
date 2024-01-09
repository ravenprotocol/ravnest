# Inference Script

import numpy as np
import torch

submodel_0 = torch.jit.load('trained_submodels/submod_0.pt')
submodel_1 = torch.jit.load('trained_submodels/submod_1.pt')
submodel_2 = torch.jit.load('trained_submodels/submod_2.pt')

test_input = np.array([2, 2, 1, 2, 0, 2])

def forward(inp):
    submodel_0.eval()
    submodel_1.eval()
    submodel_2.eval()
    with torch.no_grad():
        out_0 = submodel_0(inp)
        out_1 = submodel_1(*out_0)
        out_2 = submodel_2(*out_1)

    return out_2

def generate(x, steps=6):
    x = np.expand_dims(x,axis=0)
    x = torch.tensor(x)
    with torch.no_grad():
        for i in range(6):
            out = forward(x)
            out_tensor = out[:,-1,:]
            out_token = torch.argmax(out_tensor, axis=-1, keepdim=True)
            x = torch.concat((x, out_token), axis=-1)

    print('Sorted Prediction: ', x[:,6:])


generate(test_input, 6)
