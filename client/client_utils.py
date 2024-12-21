import random
import scipy
import numpy as np
from DataSampling import LocalDataset
import torch.nn as nn
import torch
from llg import get_label_stats,get_emb,post_process_emb,get_irlg_res


def estimate_static_RLU_with_posterior(args, N, mu, new_mu, O):
    max_diff = 100
    count = 0
    unit = args.local_epochs
    last_epoch = args.local_epochs - 1
    while max_diff >= 0.1 and count < 5:
        count += 1
        n = [round(i / args.local_epochs) for i in N]

        new_shift = []
        new_shift_softmax = []
        new_shift_softmax.append(scipy.special.softmax(mu))
        new_shift.append(mu)

        for t in range(args.local_epochs - 1):
            gb = np.zeros(args.n_classes)
            for i in range(args.n_classes):
                gb[i] = -n[i] / args.batch_size + scipy.special.softmax(new_shift[-1])[i]
            latent_dim = len(O)
            Delta = np.zeros(args.n_classes)
            for i in range(args.n_classes):
                sum_delta = 0
                for d in range(latent_dim):
                    sum_delta += -args.lr * gb[i] * O[d] * O[d]
                Delta[i] = sum_delta
            new_shift.append(new_shift[-1] + Delta)
            new_shift_softmax.append(scipy.special.softmax(new_shift[-1]))
        Diff = new_shift[last_epoch] - new_mu[last_epoch]

        larger = np.where(new_shift[last_epoch] - new_mu[last_epoch] >= 0)[0]
        abs_larger = abs(Diff[larger])
        smaller = np.where(new_shift[last_epoch] - new_mu[last_epoch] < 0)[0]
        abs_smaller = abs(Diff[smaller])

        if len(larger.tolist()) == 0 or len(smaller.tolist()) == 0:
            break

        max_diff = max(np.max(abs_larger), np.max(abs_smaller))

        idx_max_larger = np.argmax(abs_larger)
        idx_max_larger_N = larger[idx_max_larger]

        cnt = 0
        while N[idx_max_larger_N] < unit and cnt < 10:
            abs_larger[idx_max_larger] = 0
            idx_max_larger = np.argmax(abs_larger)
            idx_max_larger_N = larger[idx_max_larger]
            cnt += 1

        N[idx_max_larger_N] = N[idx_max_larger_N] - unit

        idx_max_smaller = np.argmax(abs_smaller)
        idx_max_smaller_N = smaller[idx_max_smaller]
        N[idx_max_smaller_N] = N[idx_max_smaller_N] + unit

    return np.mean(new_shift_softmax, axis=0)

def estimate_static_RLU(args, model, aux_dataset):

    model.train()
    aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=args.batch_size, shuffle=True)

    model.train()
    predictions = []
    predictions_softmax = []
    ground_truths = []
    for batch_idx, (inputs, targets) in enumerate(aux_loader):
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


def estimate_static_LLG(args, model,aux_data):
    impact = 0
    offset = torch.zeros(args.n_classes)
    label_dict = {}
    if args.dataset == 'SVHN':
        y_aux = np.array(aux_data.dataset.labels)
    else:
        y_aux = np.array(aux_data.dataset.targets)
    K = args.n_classes
    for k in range(K):
        idx_k = np.where(y_aux == k)[0]
        label_dict[k] = list(idx_k)

    model.train()
    criterion = nn.CrossEntropyLoss()
    K = args.n_classes
    g_bar = 0
    prop = args.prop
    for k in range(K):
        dict_k = label_dict[k]
        aux_num = int(prop*len(dict_k))
        aux_dict = np.random.choice(dict_k, aux_num)
        aux_dataset = LocalDataset(aux_data.dataset, aux_dict)
        aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=args.batch_size, shuffle=True)

        g_k = 0
        count = 0
        for batch_idx, (inputs, targets) in enumerate(aux_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            grads = torch.autograd.grad(loss, model.fc.parameters())
            grads = list((_.detach().cpu().clone() for _ in grads))

            w_grad, b_grad = grads[-2], grads[-1]

            gradients_for_prediction = torch.sum(w_grad, dim=-1)
            g_k += gradients_for_prediction[k]
            for j in range(K):
                if j == k:
                    continue
                else:
                    offset[j] += gradients_for_prediction[j]
            count += 1

        g_k = g_k / count
        g_bar += g_k

    impact = g_bar * (1 + 1 / args.n_classes) / (args.n_classes * args.batch_size)
    offset = offset / ((K - 1) * count)
    return impact, offset


def estimate_static_ZLG(args, model, aux_data):
    O_bar = 0
    pj = torch.zeros(args.n_classes).cuda()
    label_dict = {}

    if args.dataset == 'SVHN':
        y_aux = np.array(aux_data.dataset.labels)
    else:
        y_aux = np.array(aux_data.dataset.targets)
    K = args.n_classes
    for k in range(K):
        idx_k = np.where(y_aux == k)[0]
        label_dict[k] = list(idx_k)

    model.train()
    criterion = nn.CrossEntropyLoss()
    K = args.n_classes
    prop = args.prop

    for k in range(K):
        dict_k = label_dict[k]
        aux_num = int(prop * len(dict_k))
        aux_dict = np.random.choice(dict_k, aux_num)
        aux_dataset = LocalDataset(aux_data.dataset, aux_dict)
        aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=args.batch_size, shuffle=True)

        count = 0
        for batch_idx, (inputs, targets) in enumerate(aux_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs, embedding = model(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=-1)

            mean_probs = torch.mean(probs, dim=0)
            embedding_sum = torch.sum(embedding, dim=1)

            mean_embedding = torch.mean(embedding_sum, dim=0)

            O_bar += mean_embedding
            pj[k] += mean_probs[k]
            count += 1

    O_bar = O_bar / (args.n_classes * count)
    pj = pj / (count)
    return O_bar, pj


