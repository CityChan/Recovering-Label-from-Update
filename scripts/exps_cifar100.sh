# 1. singe local epoch with activation function relu
# iRLG
python main.py --scheme iRLG --local_epoch 1 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# RLU
python main.py --scheme RLU --local_epoch 1 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# LLGp
python main.py --scheme LLGp --local_epoch 1 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# ZLGp
python main.py --scheme ZLGp --local_epoch 1 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# 2. multiple local epochs with activation function relu
# iRLG
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# RLU
python main.py --scheme RLU --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# LLGp
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# ZLGp
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# 3. multiple local epochs with activation function relu with momentum
# iRLG
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.9 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512 --lr 0.05

# RLU
python main.py --scheme RLU --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.9 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512 --lr 0.05

# LLGp
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.9 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512 --lr 0.05

# ZLGp
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.9 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512 --lr 0.05


# 4. multiple local epochs with activation function relu with fedprox
# iRLG
python main.py --scheme iRLG --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# RLU
python main.py --scheme RLU --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# LLGp
python main.py --scheme LLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512

# ZLGp
python main.py --scheme ZLGp --local_epoch 10 --dataset CIFAR100 --n_classes 100 --momentum 0.0 --fl_scheme fedprox --mu 0.5 --alpha 0.1 --batch_size 256 --model vgg16 --hidden 512
