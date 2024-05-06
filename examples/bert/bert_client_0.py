import torch
from ravnest.node import Node
from ravnest.trainer import Trainer
from ravnest.utils import load_node_json_configs
from torch.utils.data import DataLoader
# from torchvision import transforms
import numpy as np
import random
from transformers import BertTokenizerFast

random.seed(42)
torch.manual_seed(42)
# torch.manual_seed_all(42)
torch.random.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':

    node_name = 'node_0'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer=torch.optim.Adam
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
    # input_ids = torch.randint(1,11,size=(32,10))

    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer,
                device=torch.device('cuda'),
                **node_metadata
                )
    
    node.no_grad_forward_compute(tensors=input_ids, output_type='no_grad_forward')

    # trainer = Trainer(node=node,
    #                   train_loader=train_loader,
    #                   val_loader=val_loader,
    #                   val_freq=64,
    #                   epochs=100,
    #                   batch_size=64,
    #                   inputs_dtype=torch.float32)

    # trainer.train()