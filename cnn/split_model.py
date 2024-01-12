import torch
from torch import nn
import pickle
from pip._internal.operations.freeze import freeze

from torch.fx import Tracer
from pippy.IR import Pipe
from pippy import split_into_equal_size

from torchsummary import summary

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

class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding='same')
        self.act_1 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drp_1 = nn.Dropout(0.25)
        self.bn_1 = nn.BatchNorm2d(16)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.act_2 = nn.ReLU()
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drp_2 = nn.Dropout(0.25)
        self.bn_2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(in_features=32,out_features=256)
        self.act_3 = nn.ReLU()
        self.drp_3 = nn.Dropout(0.4)
        self.bn_3 = nn.BatchNorm1d(256)
        self.dense_2 = nn.Linear(in_features=256, out_features=10)
        self.act_4 = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.act_1(out)
        out = self.maxpool2d_1(out)
        out = self.drp_1(out)
        out = self.bn_1(out)
        # print(out.shape)

        out = self.maxpool2d_2(out)
        # print(out.shape)

        out = self.conv2d_2(out)
        out = self.act_2(out)
        out = self.maxpool2d_3(out)
        # print(out.shape)

        out = self.drp_2(out)
        out = self.bn_2(out)
        # print(out.shape)
        out = self.flatten(out)
        out = self.dense_1(out)
        out = self.act_3(out)
        out = self.drp_3(out)
        out = self.bn_3(out)
        out = self.dense_2(out)
        out = self.act_4(out)
        return out
    
model = Net()

# summary(model, input_data=(1,8,8))
# model(torch.tensor(np.random.randn(32,1,8,8), dtype=torch.float32))
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
        with open('cnn/templates/{}.pkl'.format(key), 'wb') as file:
            pickle.dump(value,file)
    else:
        with open('cnn/templates/{}_output.pkl'.format(key), 'wb') as file:
            pickle.dump(value,file)

for key, value in compiled_input_dict.items():
    with open('cnn/templates/{}_input.pkl'.format(key), 'wb') as file:
        pickle.dump(value,file)
        


for key, val in pipe.split_gm._modules.items():
    script = torch.jit.script(val)
    script.save('cnn/{}.pt'.format(key))
#     print(key, val)