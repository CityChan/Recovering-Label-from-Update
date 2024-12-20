import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import os




def mnist_get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.train_labels[row] == digit:
            shard.append(ds.train_data[row].numpy())
        row += 1

    return row, shard


def cifar_get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.targets[row] == digit:
            shard.append(ds.data[row])
        row += 1

    return row, shard



def create_MNIST_ds_1shard_per_client(n_clients, samples_train, samples_test):

    MNIST_train = datasets.MNIST(root="./data", train=True, download=True)
    MNIST_test = datasets.MNIST(root="./data", train=False, download=True)

    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(int(n_clients/10)):
            row_train, shard_train = mnist_get_1shard(
                MNIST_train, row_train, i, samples_train
            )
            row_test, shard_test = mnist_get_1shard(
                MNIST_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]
    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = "./data/data_partitions/"
    train_path = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)




def create_CIFAR10_ds_1shard_per_client(n_clients, samples_train, samples_test):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         transforms.Resize((224,224))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    CIFAR10_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    CIFAR10_test =  datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(int(n_clients/10)):
            row_train, shard_train = cifar_get_1shard(
                CIFAR10_train, row_train, i, samples_train
            )
            row_test, shard_test = cifar_get_1shard(
                CIFAR10_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]
    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train
        
    folder = "./data/data_partitions/"
    train_path = f"CIFAR10_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"CIFAR10_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)
        



def get_dataloaders_shard(dataset, n_clients, batch_size: int, shuffle=True):

    folder = "./data/"

    if dataset == "MNIST_shard":
        samples_train, samples_test = int(50000/n_clients), int(8000/n_clients)

        file_name_train = f"data_partitions/MNIST_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"data_partitions/MNIST_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = folder + file_name_test
        
        file_name_labels = f"data_partitions/MNIST_shard_train_{n_clients}_{samples_train}_labels.pkl"
        path_labels = folder + file_name_labels
        
        if not os.path.isfile(path_train):
            create_MNIST_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    elif dataset == "CIFAR10_shard":
        samples_train, samples_test = int(50000/n_clients), int(10000/n_clients)

        file_name_train = f"data_partitions/CIFAR10_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"data_partitions/CIFAR10_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = folder + file_name_test
        
        file_name_labels = f"data_partitions/CIFAR10_shard_train_{n_clients}_{samples_train}_labels.pkl"
        path_labels = folder + file_name_labels
        
        if not os.path.isfile(path_train):
            create_CIFAR10_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    
    
    if not os.path.isfile(path_labels):
        Counts = count_data_partitions(list_dls_train, dataset[:-6])
        with open(path_labels, "wb") as output:
            pickle.dump(Counts, output)
    with open(path_labels, "rb") as r:
        Counts = pickle.load(r)
        print('Print out data label distributions for each clients:')
        for idx in range(len(Counts)):
            print(Counts[idx])
            
    return list_dls_train, list_dls_test

def clients_set_shard(file_name, n_clients, batch_size=64, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = ShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl



def count_data_partitions(list_loaders, dataset):
    n_clients = len(list_loaders)
    Counts = []
    if dataset == 'MNIST' or 'CIFAR10' or 'SVHN':
        n_classes = 10
    if dataset == 'CIFAR100':
        n_classes = 100
    if dataset == 'TinyImageNet':
        n_classes = 200
    for idx in range(n_clients):
        counts = [0]*n_classes
        for batch_idx,(X,y) in enumerate(list_loaders[idx]):
            batch_size = len(y)
            y = np.array(y)
            for i in range(batch_size):
                counts[int(y[i])] += 1
        Counts.append(counts)
    return Counts



class ShardDataset(Dataset):
    """Convert the pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open(file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = np.vstack(dataset[0][k])

            vector_labels = list()
            for idx, digit in enumerate(dataset[1][k]):
                vector_labels += [digit] * len(dataset[0][k][idx])

            self.labels = np.array(vector_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        x = torch.Tensor([self.features[idx]]) / 255
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y



class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
    
def LocalDataloaders(dataset, dict_users, batch_size, Shuffle = True):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,dict_users[i]),
                        batch_size=batch_size,
                        shuffle = Shuffle,
                        drop_last=True)
        loaders.append(loader)
    return loaders


def get_dataloaders_Dirichlet(n_clients, alpha=0.5,rand_seed = 0, dataset = 'CIFAR10', batch_size = 64):
    if dataset == 'CIFAR10':
        K = 10
        data_dir = './data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
        file_name_train = f"data_partitions/CIFAR10_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train

        file_name_test = f"data_partitions/CIFAR10_test_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_test = data_dir + file_name_test
        
        file_name_labels = f"data_partitions/CIFAR10_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
                
        
    if dataset == 'CIFAR100':
        K = 100
        data_dir = './data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
  
        file_name_train = f"data_partitions/CIFAR100_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train

        file_name_test = f"data_partitions/CIFAR100_test_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_test = data_dir + file_name_test
        
        file_name_labels = f"data_partitions/CIFAR100_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
        
    if dataset == 'MNIST':
        K = 10
        data_dir = './data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
        file_name_train = f"data_partitions/MNIST_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train

        file_name_test = f"data_partitions/MNIST_test_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_test = data_dir + file_name_test
        
        file_name_labels = f"data_partitions/MNIST_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
        
    if dataset == 'SVHN':
        K = 10
        data_dir = './data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.labels)
        y_test = np.array(test_dataset.labels)
        
        file_name_train = f"data_partitions/SVHN_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train

        file_name_test = f"data_partitions/SVHN_test_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_test = data_dir + file_name_test
        
        file_name_labels = f"data_partitions/SVHN_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
        
    if dataset == 'TinyImageNet':
        K = 200
        data_dir = './data/'
        train_data_path = '../data/tiny-imagenet-200/train'
        test_data_path = '../data/tiny-imagenet-200/val'

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_dataset = ImageFolder(root=train_data_path, transform=transform)
        test_dataset =  ImageFolder(root=test_data_path, transform=transform)
        
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
        file_name_train = f"data_partitions/TinyImageNet_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train

        file_name_test = f"data_partitions/TinyImageNet_test_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_test = data_dir + file_name_test
        
        file_name_labels = f"data_partitions/TinyImageNet_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
        
        
    if not os.path.isfile(path_train):
        
        min_size = 0
        N = len(train_dataset)
        N_test = len(test_dataset)
        net_dataidx_map = {}
        net_dataidx_map_test = {}
        np.random.seed(rand_seed)

        while min_size < 0.1*(N/n_clients):
            idx_batch = [[] for _ in range(n_clients)]
            idx_batch_test = [[] for _ in range(n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                idx_k_test = np.where(y_test == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
                ## Balance
                proportions_train = np.array([p*(len(idx_j)<N/n_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions_test = np.array([p*(len(idx_j)<N_test/n_clients) for p,idx_j in zip(proportions,idx_batch_test)])
                proportions_train = proportions_train/proportions_train.sum()
                proportions_test = proportions_test/proportions_test.sum()
                proportions_train = (np.cumsum(proportions_train)*len(idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions_test)*len(idx_k_test)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch_test,np.split(idx_k_test,proportions_test))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            net_dataidx_map_test[j] = idx_batch_test[j]
   
    
        with open(path_train, "wb") as output:
            pickle.dump(net_dataidx_map, output)

        with open(path_test, "wb") as output:
            pickle.dump(net_dataidx_map_test, output)
            
    with open(path_train, "rb") as r:
        dict_users = pickle.load(r)
        Loaders_train = LocalDataloaders(train_dataset,dict_users,batch_size,Shuffle = True)
        
    with open(path_test, "rb") as r:
        dict_users_test = pickle.load(r)
        Loaders_test = LocalDataloaders(test_dataset,dict_users_test,batch_size,Shuffle = True)
        
    if not os.path.isfile(path_labels):
        Counts = count_data_partitions(Loaders_train, dataset)
        with open(path_labels, "wb") as output:
            pickle.dump(Counts, output)
    with open(path_labels, "rb") as r:
        Counts = pickle.load(r)
        print('Print out data label distributions for each clients:')
        for idx in range(len(Counts)):
            print(Counts[idx])
            
            
    return Loaders_train, Loaders_test