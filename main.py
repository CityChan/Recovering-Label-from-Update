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
from models import get_model
from option import args_parser
from utils import average_weights,global_acc,AverageMeter
from llg import get_label_stats,get_emb,post_process_emb,get_irlg_res
args = args_parser()
print(args)