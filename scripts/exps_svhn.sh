# 1. singe local epoch with activation function relu
# iRLG
python main.py --scheme iRLG --local_epoch 1 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400

# RLU
python main.py --scheme RLU --local_epoch 1 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400

# LLGp
python main.py --scheme LLGp --local_epoch 1 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400

# ZLGp
python main.py --scheme ZLGp --local_epoch 1 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400

# 2. multiple local epochs with activation function relu
# iRLG
python main.py --scheme iRLG --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400

# RLU
python main.py --scheme RLU --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400

# LLGp
python main.py --scheme LLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400

# ZLGp
python main.py --scheme ZLGp --local_epoch 10 --dataset SVHN --momentum 0.0 --alpha 0.1 --batch_size 32 --model lenet5 --hidden 400