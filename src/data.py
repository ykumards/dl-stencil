from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms


def setup_cifar10_dataloaders(batch_size=128, download=True):
    root = '../dataset'
    trans = transforms.ToTensor()
    train_set = dset.CIFAR10(
        root=root, train=True, transform=trans,
        download=download
    )
    test_set = dset.CIFAR10(
        root=root, train=False, transform=trans
    )

    kwargs = {
        'num_workers': 4
    }

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, **kwargs
    )
    return train_loader, test_loader
