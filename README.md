# Recovering-Label-from-Update
This is the official implementation of the ICML2024 paper "Recovering Labels from Local Updates in Federated Learning".

## Requirements
This code is implemented in Pytorch and Cuda. We ran all experiments in the virtual environment created by Conda.
For installing the virtual environment:
```
conda create -n RLU python=3.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Dataset Preparing:
* Create directory `mkdir data` and subdirectory `mkdir data/data_partitions`
* **SVHN/CIFAR10/CIFAR100**: should automatically be downloaded
* **TinyImageNet**: download dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip.
  
After unzipping, place them in `data/` directory

## Project Structure
* `option.py`: this file contains configuartions for all schemes. 
* `DataSampling.py`: this file is relevant to data partions for federated learning
* `models.py`: this file define models for different experiments. Please change the activation functions if needed.
* `utils.py`: general utilization 
* `llg.py`: this is adapted from the attack scheme [LLG](https://github.com/tklab-tud/LLG)
* **./client**: for each method, we defined adaptive client objects. We currently only release FedAvg and FedProx. FedDyn and Scaffold will be soon.
* `./client/client_utils.py`: utilization for the attacking schemes


## Run Attacks:

#### For SVHN with multiple local epochs

```
python main.py --scheme iRLG --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme RLU --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme LLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
python main.py --scheme ZLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.5 --batch_size 32 --model lenet5 --hidden 400
```

#### For CIFAR10 with multiple local epochs

```
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
python main.py --scheme RLU --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR10 --momentum 0.0 --alpha 0.5 --batch_size 64 --model vgg16 --hidden 512
```

#### For CIFAR100 with multiple local epochs
```
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py --scheme RLU --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
```

#### For 10-split ImageNet-R

```
python main.py --device "4" --config ./configs/mimg10_duallora.json 
```

#### For 20-split ImageNet-R

```
python main.py --device "5" --config ./configs/mimg20_duallora.json 
```
