# Note: Place the tiny-imagenet-200 Dataset in home directory
import torch
import time
from sklearn import datasets
from ravnest.node import Node
from ravnest.trainer import Trainer
from ravnest.utils import load_node_json_configs
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time

def get_dataset(root=None):

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomResizedCrop(64),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=root, transform=transform)
    subset_indices = range(1000)
    subset_dataset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=0)
    return train_loader

train_loader = get_dataset(root='./tiny-imagenet-200/train')

if __name__ == '__main__':

    node_name = 'node_0'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer=torch.optim.Adam
    
    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer,
                **node_metadata
                )
        
    trainer = Trainer(node=node,
                      train_loader=train_loader,
                      epochs=1,
                      batch_size=64,
                      step_size=64)
    
    t1 = time.time()

    trainer.train()
    
    print('Training Done!: ', time.time() - t1)
