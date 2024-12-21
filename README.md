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
* Create directory `mkdir data`
* **SVHN/CIFAR10/CIFAR100**: should automatically be downloaded
* **TinyImageNet**: download dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip.
  
After unzipping, place them in `data/` directory

## Project Structure
* **configs**: this directory contains configuartions of training for all schemes. All parameteres are stored in `.json` file.
* **data**: put all datasets in this directory.
* **dataloaders**: only for spliting test set of **DomainNet**.
* **logs**: storing the training and test results.
* **methods**: this directory contains all methods we used in the experiments.
* **models**: for each method, we defined adaptive model structure.
* **utils**: this directory contain utilization functions needed in the traing including: creating continual learning data partitions, computing accuracy.


## Run Training:
#### For 10-split CIFAR100
```
python main.py --device "0" --config ./configs/cifar100_duallora.json 
```

#### For 10-split Tiny-ImageNet

```
python main.py --device "1" --config ./configs/tinyimage10_duallora.json 
```

#### For 20-split Tiny-ImageNet

```
python main.py --device "2" --config ./configs/tinyimage20_duallora.json 
```

#### For 5-split ImageNet-R
```
python main.py --device "3" --config ./configs/mimg5_duallora.json 
```

#### For 10-split ImageNet-R

```
python main.py --device "4" --config ./configs/mimg10_duallora.json 
```

#### For 20-split ImageNet-R

```
python main.py --device "5" --config ./configs/mimg20_duallora.json 
```
