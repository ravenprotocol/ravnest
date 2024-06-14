import torch
from ravnest import Node, Trainer, set_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

set_seed(42)

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

    node = Node(name = 'node_0', 
                optimizer = torch.optim.SGD,
                optimizer_params = {'lr':0.01, 'momentum':0.9, 'weight_decay':0.0005},
                device=torch.device('cpu')
                )
        
    trainer = Trainer(node=node,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      val_freq=20,
                      epochs=10,
                      batch_size=64,
                      step_size=64,
                      save=True)

    trainer.train()
