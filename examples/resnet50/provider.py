# Note: Place the tiny-imagenet-200 Dataset in home directory
import torch
from ravnest import Node, Trainer, set_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from TinyImageNet import TinyImageNet

set_seed(42)

def get_dataset(root=None):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    training_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    in_memory = False

    generator = torch.Generator()
    generator.manual_seed(42)

    training_set = TinyImageNet(root, 'train', transform=training_transform, in_memory=in_memory)
    # train_subset = Subset(training_set, range(1000))
    train_loader = DataLoader(training_set, batch_size=100, shuffle=True, generator=generator, num_workers=0)


    val_set = TinyImageNet(root, 'val', transform=valid_transform, in_memory=in_memory)
    val_loader = DataLoader(val_set, batch_size=100, shuffle=False, num_workers=0)

    return train_loader, val_loader

train_loader, val_loader = get_dataset(root='./tiny-imagenet-200')

if __name__ == '__main__':

    node = Node(name = 'node_0',
                optimizer = torch.optim.SGD,
                optimizer_params = {'lr':0.01, 'momentum':0.9, 'weight_decay':0.0005},
                criterion = torch.nn.CrossEntropyLoss(), 
                labels = train_loader,
                test_labels = val_loader,
                device=torch.device('cuda')
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