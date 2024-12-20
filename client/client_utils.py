import random
import scipy
import numpy as np
from DataSampling import LocalDataset
import torch.nn as nn
import torch
from llg import get_label_stats,get_emb,post_process_emb,get_irlg_res

def estimate_static_RLU(args, model, aux_dataset):

    model.train()
    criterion = nn.CrossEntropyLoss()

    aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=args.batch_size, shuffle=True)

    model.train()
    predictions = []
    predictions_softmax = []
    ground_truths = []
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(aux_loader):
        labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, args.n_classes)
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, _ = model(inputs)
        probs = torch.softmax(outputs, dim=-1)
        ground_truths.append(np.array(targets.detach().cpu()))
        predictions.append(np.array(outputs.detach().cpu()))
        predictions_softmax.append(np.array(probs.detach().cpu()))

    mis_predictions_maxrix = matrix(args,predictions, ground_truths)
    mis_predictions_softmax = matrix(args, predictions_softmax, ground_truths)
    mu = np.zeros(args.n_classes)
    for i in range(args.n_classes):
        mu[i] = (np.sum(mis_predictions_maxrix[i]) - mis_predictions_maxrix[i, i]) / (args.n_classes - 1)
    shift = np.zeros(args.n_classes)
    for i in range(args.n_classes):
        shift[i] = (np.sum(mis_predictions_softmax[i]) - mis_predictions_softmax[i, i]) / (args.n_classes - 1)
    return mu, shift



def estimated_entropy_from_grad(args, shift, bias, B):
    n = args.n_classes
    solution = [0] * n

    bias = -np.array(bias)

    n = [0] * args.n_classes
    for i in range(args.n_classes):
        bias[i] = bias[i] + shift[i]

    for i in range(args.n_classes):
        if bias[i] < 0:
            bias[i] = 0

    s = np.sum(abs(bias))
    for i in range(args.n_classes):
        bias[i] = bias[i] / s
        solution[i] = round(bias[i] * B)

    return solution


def learn_stat_vector(args, n, predictions, ground_truths):
    mis_predictions = []
    for i in range(len(predictions) - 1):
        for j in range(args.batch_size):
            if ground_truths[i][j] == n:
                mis_predictions.append(predictions[i][j])

    if len(mis_predictions) == 0:
        mis_predictions.append(0)

    return np.array(mis_predictions)


def learn_stat(args, k, n, predictions, ground_truths):
    mis_predictions = []
    for i in range(len(predictions) - 1):
        for j in range(args.batch_size):
            if ground_truths[i][j] == n:
                mis_predictions.append(predictions[i][j][k])

    if len(mis_predictions) == 0:
        mis_predictions.append(0)

    return np.array(mis_predictions)


def matrix(args, predictions, ground_truths):
    mis_predictions_maxrix = np.zeros((args.n_classes, args.n_classes))
    for i in range(args.n_classes):
        for j in range(args.n_classes):
            mis_predictions_maxrix[i][j] = np.mean(learn_stat(args, i, j, predictions, ground_truths))

    return mis_predictions_maxrix


def matrix_mean_var(args, predictions, ground_truths):
    mis_predictions_maxrix = np.zeros((args.n_classes, args.n_classes))
    for i in range(args.n_classes):
        stat = learn_stat_vector(i, predictions, ground_truths)
        if len(stat) == 1:
            mis_predictions_maxrix[:, i] = 0
            continue
        mean = np.mean(stat, axis=0)
        cov = np.cov(np.transpose(stat))
        samples = np.random.multivariate_normal(mean, cov, size=10000)
        softmax_samples = scipy.special.softmax(samples, axis=1)
        mean_softmax = np.mean(softmax_samples, axis=0)
        mis_predictions_maxrix[:, i] = mean_softmax
    return mis_predictions_maxrix