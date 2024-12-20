from __future__ import print_function, absolute_import
import torch
import copy

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        m,n = correct[:k].shape
        correct_k = correct[:k].reshape((1, m*n)).float().flatten().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def average_weights(w):
    """
    average the weights from all local models
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def sum_list(a, j):
    b = 0
    for i in range(len(a)):
        if i != j:
            b += a[i]
    return b


def global_acc(global_model, Loaders_test):
    acc_top1 = 0
    acc_top5 = 0
    for batch_idx, (inputs, targets) in enumerate(Loaders_test):
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        outputs,_ = global_model(inputs)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        acc_top1 += prec1
        acc_top5 += prec5
    acc_top1 = acc_top1/ len(Loaders_test)
    acc_top5 = acc_top5/ len(Loaders_test)
    return acc_top1, acc_top5


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count