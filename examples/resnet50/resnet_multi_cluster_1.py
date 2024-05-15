# Note: Place the tiny-imagenet-200 Dataset in home directory
import torch
from ravnest.node import Node
from ravnest.trainer import Trainer
from ravnest.utils import load_node_json_configs
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from TinyImageNet import TinyImageNet
import numpy as np
import random

random.seed(42)
torch.manual_seed(42)
# torch.manual_seed_all(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

def get_dataset(root=None):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    training_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    in_memory = False

    generator = torch.Generator()
    generator.manual_seed(7)

    training_set = TinyImageNet(root, 'train', transform=training_transform, in_memory=in_memory)
    # train_subset = Subset(training_set, range(1000))
    train_loader = DataLoader(training_set, batch_size=16, shuffle=True, generator=generator, num_workers=0)


    val_set = TinyImageNet(root, 'val', transform=valid_transform, in_memory=in_memory)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    return train_loader, val_loader

train_loader, val_loader = get_dataset(root='./tiny-imagenet-200')

if __name__ == '__main__':

    node_name = 'node_0'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer = torch.optim.SGD
    optimizer_params = {'lr':0.01, 'momentum':0.9, 'weight_decay':0.0005}

    node = Node(name = node_name,
                model = model,
                optimizer = optimizer,
                optimizer_params = optimizer_params,
                device=torch.device('cuda'),
                **node_metadata
                )

    trainer = Trainer(node=node,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      val_freq=100,
                      epochs=5,
                      batch_size=100,
                      step_size=100,
                      save=True)

    trainer.train()
    trainer.evaluate()