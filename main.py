import os
import torch
import numpy as np
import time
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from models import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from DataSampling import get_dataloaders_shard,count_data_partitions,get_dataloaders_Dirichlet,LocalDataset
from option import args_parser
args = args_parser()
print(args)

# Generating data partitions based on Dirichlet distribution
Loaders_train,Loaders_test = get_dataloaders_Dirichlet(n_clients = args.n_clients, alpha=args.alpha, rand_seed = 0,
                                                       dataset = args.dataset, batch_size = args.batch_size)


K = int(args.sampling_rate*args.n_clients)
data_dir = './data/'
apply_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if args.dataset == 'SVHN':
    test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                 transform=apply_transform)
    total_train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                        transform=apply_transform)

if args.dataset == 'CIFAR10':
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    total_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                      transform=apply_transform)
if args.dataset == 'CIFAR100':
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                     transform=apply_transform)
    total_train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                            transform=apply_transform)

if args.dataset == 'Tiny':
    test_data_path = './data/tiny-imagenet-200/val'
    train_data_path = './data/tiny-imagenet-200/train'

    test_dataset = ImageFolder(root=test_data_path, transform=apply_transform)
    total_train_dataset = ImageFolder(root=train_data_path, transform=apply_transform)

validate_num = len(test_dataset)
args.prop = 0.1
whole_range = range(validate_num)
aux_num = int(args.prop*validate_num)
aux_dict = np.random.choice(whole_range, aux_num)
aux_dataset = LocalDataset(test_dataset, aux_dict)
loader_test = torch.utils.data.DataLoader(aux_dataset, batch_size=args.batch_size,shuffle=True)
total_loader_train = torch.utils.data.DataLoader(total_train_dataset, batch_size=args.batch_size,shuffle=True)

# create global model
channel = 3
global_model = get_model(model_name=args.model,
                         net_params=(args.n_classes, channel, args.hidden),
                         device=device,
                         n_hidden=1,
                         n_dim=300,
                         batchnorm=False,
                         dropout=True,
                         tanh=False,
                         leaky_relu=False).cuda()

global_weights = global_model.state_dict()