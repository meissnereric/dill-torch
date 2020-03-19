import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def get_fake_dataloaders(size=10, batch_size=2, image_size=(224), num_classes=10):

    train_dataset = dsets.FakeData(size=size, image_size=image_size,
        num_classes=num_classes, random_offset=1)

    test_dataset = dsets.FakeData(size=size, image_size=image_size,
        num_classes=num_classes, random_offset=10)

    dataloaders = _make_dataloaders(train_dataset, test_dataset, batch_size)
    return dataloaders

def get_cifar_dataloaders(batch_size):
    data_transforms = transforms.Compose([
                                          transforms.Resize((224, 224)),
                                          transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = dsets.CIFAR10(root='./test_data/cifar',
                               train=True,
                               transform=data_transforms,
                               download=True)

    test_dataset = dsets.CIFAR10(root='./test_data/cifar',
                               train=False,
                               transform=data_transforms,
                               download=True)

    dataloaders = _make_dataloaders(train_dataset, test_dataset, batch_size)
    return dataloaders

def _make_dataloaders(train_dataset, test_dataset, batch_size):

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        dataloaders = {'train': train_loader, 'val': test_loader}
