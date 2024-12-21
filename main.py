import os
import torch
import numpy as np
import time
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from models import get_model
from client.client_base import Client
import random
from utils import accuracy,average_weights,sum_list,global_acc

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

random.seed(42)
label_dict = {}
y_aux = np.array(test_dataset.targets)
K = args.n_classes
for k in range(K):
    idx_k = np.where(y_aux == k)[0]
    label_dict[k] = list(idx_k)

prop = args.prop
aux_dict = []
for k in range(K):
    dict_k = label_dict[k]
    aux_num = int(prop * len(dict_k))
    aux_dict.append(np.random.choice(dict_k, aux_num))

aux_dict = np.concatenate(aux_dict)
aux_dataset = LocalDataset(test_dataset, aux_dict)
total_loader_train = torch.utils.data.DataLoader(total_train_dataset, batch_size=args.batch_size,shuffle=True)

# create global model
channel = 3
tanh = False
if args.activation == 'tanh':
    tanh = True
global_model = get_model(model_name=args.model,
                         net_params=(args.n_classes, channel, args.hidden),
                         device=device,
                         n_hidden=1,
                         n_dim=300,
                         batchnorm=False,
                         dropout=True,
                         tanh=tanh,
                         leaky_relu=False).cuda()

global_weights = global_model.state_dict()
print("==> creating models")
Clients = []
for idx in range(args.n_clients):
    Clients.append(Client(args, Loaders_train[idx], idx, device, args.model, aux_dataset))

# Attacking after training the global model
# args.epochs = 0
# for epoch in range(args.epochs):
#     local_weights = []
#     clients_models = []
#     global_model.train()
#     print(f'\n | Global Training Round : {epoch+1} |\n')
#
#     sampled_clients = np.random.choice(args.n_clients, K , replace=False)
#     for idx in sampled_clients:
#         Clients[idx].load_model(global_weights)
#         Clients[idx].local_training(epoch)
#         local_weights.append(Clients[idx].model.state_dict())
#         clients_models.append(copy.deepcopy(Clients[idx].model))
#
#     global_weights = average_weights(local_weights)
#     global_model.load_state_dict(global_weights)
#     acc_top1, acc_top5 = global_acc(global_model, total_loader_train)
#     print(f'Top 1 training accuracy: {acc_top1}, Top 5 accuracy: {acc_top5}  at global round {epoch}.')

cAcc = []
iAcc = []
for idx in range(args.n_clients):
    print('client: ', idx)
    Clients[idx].load_model(global_weights)
    if args.scheme == 'iRLG':
        acc1, acc2 = Clients[idx].iRLG(global_weights)
    if args.scheme == 'RLU':
        acc1, acc2 = Clients[idx].RLU(global_weights)
    if args.scheme == 'LLGp':
        acc1, acc2 = Clients[idx].LLGp(global_weights)
    if args.scheme == 'ZLGp':
        acc1, acc2 = Clients[idx].ZLGp(global_weights)
    cAcc.append(acc1)
    iAcc.append(acc2)
average_cAcc = np.mean(np.array(cAcc))
average_iAcc = np.mean(np.array(iAcc))

print('average cAcc: ', average_cAcc)
print('average iAcc: ', average_iAcc)
