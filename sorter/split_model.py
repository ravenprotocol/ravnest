from mingpt.model_without_padding_mask import GPT
import torch
from torch.utils.data import Dataset
import pickle
import torch
from torch.fx import Tracer
import numpy as np
import _pickle as cPickle
import random
from pippy.IR import annotate_split_points, PipeSplitWrapper, Pipe
from pippy import split_into_equal_size

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

class CustomTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        
        if hasattr(m, '_is_leaf_module') and m._is_leaf_module:
            return True

        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

def split_model(model, n_splits=3):
    custom_tracer = CustomTracer()
    split_policy = split_into_equal_size(n_splits)
    pipe = Pipe.from_tracing(model, tracer=custom_tracer, split_policy=split_policy)
    return pipe

# print an example instance of the dataset
train_dataset = SortDataset('train')
test_dataset = SortDataset('test')

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

pipe = split_model(model, 3)



def get_arg_index(name, submod_args):
    for i in range(len(submod_args)):
        if submod_args[i].name == name:
            return i
    return -1

compiled_input_dict = {}
compiled_output_dict = {'model_inputs':{}}

for node in pipe.split_gm.graph.nodes:
    print(node.name, node.args)
    if 'submod' in node.name:
        input_dict = {}

        compiled_output_dict[node.name] = {}
        if node.name == 'submod_0':
            submod_0_args = node.args
            for i in range(len(submod_0_args)):
                compiled_output_dict['model_inputs'][i] = {}
        else:
            if len(node.args) > 0:
                for i in range(len(node.args)):
                    input_dict[i] = {}
            arg_index = 0
            for arg in node.args:
                if 'submod' in arg.name:
                    input_dict[arg_index][arg.name] = 'placeholder:tensor'

                    if compiled_output_dict[arg.name].get(arg_index, None) is not None:
                        compiled_output_dict[arg.name][arg_index]['target'].append(node.name)   
                    else:
                        compiled_output_dict[arg.name][arg_index] = {'target' : [node.name]}


                elif 'getitem' in arg.name:
                    inner_arg = arg.args             
                    input_dict[arg_index][inner_arg[0].name] = inner_arg[1]

                    if compiled_output_dict[inner_arg[0].name].get(inner_arg[1], None) is not None:
                        compiled_output_dict[inner_arg[0].name][inner_arg[1]]['target'].append(node.name)   
                    else:
                        compiled_output_dict[inner_arg[0].name][inner_arg[1]] = {'target' : [node.name]}

                
                else:
                    index = get_arg_index(arg.name, submod_0_args)
                    input_dict[arg_index]['model_inputs'] = index

                    if compiled_output_dict['model_inputs'][index].get('target', None) is not None:
                        compiled_output_dict['model_inputs'][index]['target'].append(node.name)
                    else:
                        compiled_output_dict['model_inputs'][index]['target'] = [node.name]


                arg_index += 1
        
        compiled_input_dict[node.name] = input_dict


print('\ncompiled input dict: ', compiled_input_dict)

print('\ncompiled output dict: ', compiled_output_dict)

for key, value in compiled_output_dict.items():
    if key == 'model_inputs':
        with open('sorter/templates/{}.pkl'.format(key), 'wb') as file:
            pickle.dump(value,file)
    else:
        with open('sorter/templates/{}_output.pkl'.format(key), 'wb') as file:
            pickle.dump(value,file)

for key, value in compiled_input_dict.items():
    with open('sorter/templates/{}_input.pkl'.format(key), 'wb') as file:
        pickle.dump(value,file)
        

for key, val in pipe.split_gm._modules.items():
    script = torch.jit.script(val)
    script.save('sorter/{}.pt'.format(key))
    # print(key, val)

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
