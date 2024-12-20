import os
import torch
import numpy as np
import time
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from DataSampling import get_dataloaders_shard,count_data_partitions,get_dataloaders_Dirichlet,LocalDataset
from option import args_parser
args = args_parser()
print(args)

# Generating data partitions based on Dirichlet distribution
Loaders_train,Loaders_test = get_dataloaders_Dirichlet(n_clients = args.n_clients, alpha=args.alpha, rand_seed = 0,
                                                       dataset = args.dataset, batch_size = args.batch_size, device = device)