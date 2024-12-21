
import argparse

def args_parser():
    
    # Optimization options

    parser = argparse.ArgumentParser(description='PyTorch Attacking Labels')
    parser.add_argument('--epochs', default= 20, type=int, metavar='N',
                        help='number of total epochs to run')
    
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='training batchsize')

    parser.add_argument('-d', '--dataset', default='CIFAR10', type=str, help='SVHN, CIFAR10, CIFAR100, Tiny-ImageNet')

    parser.add_argument('-p', '--prop', default=0.1, type=float, help='proportion of aux data')

    parser.add_argument('-m', '--model', default='resnet18', type=str, help='vgg, resnet18, resnet54, lenet')

    parser.add_argument('--activation', default='relu', type=str, help='relu, tanh, elu, selu, silu')

    parser.add_argument('--hidden', default=512, type=int, help='dimension of the hidden embeddings')

    parser.add_argument('-n', '--n_clients', default=10, type=int)
    
    parser.add_argument('--sampling_rate', default=1.0, type=float, help='sampling rate for clients')
    
    parser.add_argument('--n_classes', default = 10, type=int)

    parser.add_argument('--alpha', default=0.5, type=float)
    
    parser.add_argument('--local_epochs', default = 1, type=int, metavar='N')
    
    
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')

    parser.add_argument('--scheme', default='RLU',type=str,
                        help='RLU, iRLG, LLGp, ZLGp')

    parser.add_argument('--fl_scheme', default='fedavg', type=str,
                        help='fedavg, fedprox, feddyn, scaffold')

    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    
    parser.add_argument('-c', '--checkpoint', default='./checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint')
    
    args = parser.parse_args()
    return args

