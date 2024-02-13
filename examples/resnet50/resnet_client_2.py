# Note: Place the tiny-imagenet-200 Dataset in same directory
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

    train_dataset = datasets.ImageFolder(root=root, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    subset_indices = range(1000)
    subset_dataset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=0)
    return train_loader

train_loader = get_dataset(root='./tiny-imagenet-200/train')

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
                **node_metadata
                )
    node.start()
    while True:
        time.sleep(1)
