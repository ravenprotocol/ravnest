import torch
from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
import _pickle as cPickle
import random

"""
The following code is based on the minGPT repo by Andrej Karpathy:
https://github.com/karpathy/minGPT

Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""


class SortDataset(Dataset):
    """ 
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):
        
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that 
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unqiue digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok
        
        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1
        return x, y

split = 'train'

X_train = []
y_train = []

for i in range(51200):

    while True:
        inp = np.random.randint(0, 3, size=(6,))

        if random.uniform(0,1) < 0.5:
            if np.unique(inp).size > 6 // 2:
                continue

        h = hash(cPickle.dumps(inp.tolist()))
        inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
        if inp_split == split:
            break

    sol = np.sort(inp)
    cat = np.concatenate((inp, sol), axis=0)

    x = cat[:-1].copy()
    y = cat[1:].copy()
    y[:6-1] = -1

    X_train.append(x)   
    y_train.append(y)

X_train = np.array(X_train)
y_train = np.array(y_train)

with open('sorter/sorter_data/X_train.pkl', 'wb') as X_file:
    pickle.dump(X_train,X_file)

with open('sorter/sorter_data/y_train.pkl', 'wb') as y_file:
    pickle.dump(y_train,y_file)
