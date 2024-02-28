# Note: Place the tiny-imagenet-200 Dataset in home directory
import torch
import time
from ravnest.node import Node
from ravnest.utils import load_node_json_configs
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader

def get_dataset(root=None):

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomResizedCrop(64),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=root + '/train', transform=transform)
    train_subset_indices = range(1000)
    train_subset_dataset = Subset(train_dataset, train_subset_indices)
    train_loader = DataLoader(train_subset_dataset, batch_size=64, shuffle=False, num_workers=0)

    val_dataset = datasets.ImageFolder(root=root + '/val', transform=transform)
    val_subset_indices = range(1000, 1100)
    val_subset_dataset = Subset(val_dataset, val_subset_indices)
    val_loader = DataLoader(val_subset_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, val_loader

train_loader, val_loader = get_dataset(root='./tiny-imagenet-200')

if __name__ == '__main__':

    node_name = 'node_2'

    node_metadata = load_node_json_configs(node_name=node_name)
    model = torch.jit.load(node_metadata['template_path']+'submod.pt')
    optimizer=torch.optim.Adam
    criterion = torch.nn.functional.cross_entropy

    node = Node(name = node_name, 
                model = model, 
                optimizer = optimizer,
                criterion = criterion, 
                labels = train_loader,
                test_labels = val_loader,
                device=torch.device('cpu'),
                **node_metadata
                )

    while True:
        time.sleep(1)
