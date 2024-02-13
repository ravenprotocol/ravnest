# Note: Place the tiny-imagenet-200 Dataset in home directory
import torch
import time
from sklearn import datasets
from ravnest.node import Node
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
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    subset_indices = range(1000)
    subset_dataset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=0)
    return train_loader

train_loader = get_dataset(root='./tiny-imagenet-200/train')
test_loader = get_dataset(root='./tiny-imagenet-200/test')


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
    node.start()
    batch_size = 64
    epochs = 1
    tensor_id = 0
    n_forwards = 0

    time.sleep(3)
    t1 = time.time()
    for epoch in range(epochs):
        data_id = 0
        for images, labels in train_loader:
            # print('\nPassing batch')
            # output = model(torch.tensor(X_train, dtype=torch.float32))
            node.forward_compute(data_id=data_id, tensors=images)
            n_forwards += 1
            # print('Data id: ', data_id)
            
            data_id += 1 #images.shape[0]
            # break
            # if data_id>256:
            #     break
        print('Epoch: ', epoch)
        print('n_forward: ', n_forwards, '  node.n_backward: ', node.n_backwards)
    
    while node.n_backwards < n_forwards: #node.is_training: #
        time.sleep(1)
    
    print('Training Done!: ', time.time() - t1)

    
    pred = node.no_grad_forward_compute(tensors=next(iter(test_loader))[0], output_type='accuracy')

    
    while True:
        time.sleep(1)

