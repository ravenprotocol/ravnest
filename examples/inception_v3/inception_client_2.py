# Note: Place the tiny-imagenet-200 Dataset in home directory
import torch
import time
from ravnest.node import Node
from ravnest.utils import load_node_json_configs
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def get_dataset(root=None):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    training_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_dataset = CIFAR10(root=root, download=True, transform=training_transform)
    val_dataset = CIFAR10(root=root, train=False, transform=valid_transform)        

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, val_loader

train_loader, val_loader = get_dataset(root='./cifar10/')

if __name__ == '__main__':

    node_name = 'node_2'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer = torch.optim.SGD
    optimizer_params = {'lr':0.01, 'momentum':0.9, 'weight_decay':0.0005}

    criterion = torch.nn.CrossEntropyLoss()

    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer,
                optimizer_params = optimizer_params,
                criterion = criterion, 
                labels = train_loader,
                test_labels = val_loader,
                device=torch.device('cuda'),
                **node_metadata
                )

    while True:
        time.sleep(1)
