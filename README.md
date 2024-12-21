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
