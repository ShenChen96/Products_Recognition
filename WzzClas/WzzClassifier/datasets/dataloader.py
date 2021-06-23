import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Product_Dataloader(object):
    def __init__(self, train_dataroot, val_dataroot, unknown_dataroot, use_gpu=True, num_workers=8, batch_size=32,
                 img_size=224):
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = ImageFolder(root=train_dataroot, transform=train_transform)
        print('All Train Data:', len(trainset))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = ImageFolder(root=val_dataroot, transform=transform)
        print('All Test Data:', len(testset))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = ImageFolder(root=unknown_dataroot, transform=transform)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.num_classes = len(trainset.classes) + len(outset.classes)
        self.known = len(trainset.classes)
        self.unknown = len(outset.classes)

        print('Selected Labels: ', self.known)
        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

