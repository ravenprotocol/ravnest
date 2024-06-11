import torch
import time
import random
import numpy as np
from ravnest.node import Node
from ravnest.utils import load_node_json_configs
from torch_optimizer import Lamb
from transformers.optimization import get_linear_schedule_with_warmup

random.seed(42)
torch.manual_seed(42)
# torch.manual_seed_all(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

if __name__ == '__main__':
    
    node_name = 'node_1'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    update_frequency = 16

    optimizer=Lamb
    optimizer_params = {'lr':0.00176, 'eps':1e-6, 'debias':True, 'weight_decay':0.01, 'clamp_value':10000.0}

    scheduler = get_linear_schedule_with_warmup
    num_training_steps = 125280
    scheduler_params = {'num_warmup_steps':5000, 'num_training_steps':num_training_steps}

    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer, 
                optimizer_params = optimizer_params,
                update_frequency = update_frequency,
                lr_scheduler=scheduler,
                lr_scheduler_params=scheduler_params,
                lr_step_on_epoch_change=False,
                device=torch.device('cuda'),
                **node_metadata
                )

    while True:
        time.sleep(0)
