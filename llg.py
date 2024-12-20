from collections import Counter
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import random
def get_label_stats(gt_label, num_classes):
    LabelCounter = dict(Counter(gt_label.cpu().numpy()))
    labels = list(sorted(LabelCounter.keys()))
    existences = [1 if i in labels else 0 for i in range(num_classes)]
    num_instances = [LabelCounter[i] if i in labels else 0 for i in range(num_classes)]
    num_instances_nonzero = [item[1] for item in sorted(LabelCounter.items(), key=lambda x: x[0])]
    return labels, existences, num_instances, num_instances_nonzero
__all__ = ['accuracy']


def post_process_emb(embedding, model, device, alpha=0.01):
    embedding = embedding.to(device)
    # Feed embedding into FC-Layer to get probabilities.
    out = model.fc(embedding) * alpha
    prob = torch.softmax(out, dim=-1)
    return prob

def get_emb(grad_w, grad_b, exp_thre=10):
    # Split scientific count notation
    sc_grad_b = '%e' % grad_b
    sc_grad_w = ['%e' % w for w in grad_w]
    real_b, exp_b = float(sc_grad_b.split('e')[0]), int(sc_grad_b.split('e')[1])
    real_w, exp_w = np.array([float(sc_w.split('e')[0]) for sc_w in sc_grad_w]), \
                    np.array([int(sc_w.split('e')[1]) for sc_w in sc_grad_w])
    # Deal with 0 case
    if real_b == 0.:
        real_b = 1
        exp_b = -64
    # Deal with exponent value
    exp = exp_w - exp_b
    exp = np.where(exp > exp_thre, exp_thre, exp)
    exp = np.where(exp < -1 * exp_thre, -1 * exp_thre, exp)

    def get_exp(x):
        return 10 ** x if x >= 0 else 1. / 10 ** (-x)

    exp = np.array(list(map(get_exp, exp)))
    # Calculate recovered average embeddings for batch_i (samples of class i)
    res = (1. / real_b) * real_w * exp
    res = torch.from_numpy(res).to(torch.float32)
    return res


def iLRG(probs, grad_b, n_classes, n_images):
    # Solve linear equations to recover labels
    coefs, values = [], []
    # Add the first equation: k1+k2+...+kc=K
    coefs.append([1 for _ in range(n_classes)])
    values.append(n_images)
    # Add the following equations
    for i in range(n_classes):
        coef = []
        for j in range(n_classes):
            if j != i:
                coef.append(probs[j][i].item())
            else:
                coef.append(probs[j][i].item() - 1)
        coefs.append(coef)
        values.append(n_images * grad_b[i])
    # Convert into numpy ndarray
    coefs = np.array(coefs)
    values = np.array(values)
    # Solve with Moore-Penrose pseudoinverse
    res_float = np.linalg.pinv(coefs).dot(values)
    # Filter negative values
    res = np.where(res_float > 0, res_float, 0)
    # Round values
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        inds = np.argsort(-err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] -= 1
    elif num_mod < 0:
        inds = np.argsort(err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] += 1
    else:
        mod_res = res

    return res, mod_res


# Have Known about which labels exist
def sim_iLRG(probs, grad_b, exist_labels, n_images):
    # Solve linear equations to recover labels
    coefs, values = [], []
    # Add the first equation: k1+k2+...+kc=K
    coefs.append([1 for _ in range(len(exist_labels))])
    values.append(n_images)
    # Add the following equations
    for i in exist_labels:
        coef = []
        for j in exist_labels:
            if j != i:
                coef.append(probs[j][i].item())
            else:
                coef.append(probs[j][i].item() - 1)
        coefs.append(coef)
        values.append(n_images * grad_b[i])
    # Convert into numpy ndarray
    coefs = np.array(coefs)
    values = np.array(values)
    # Solve with Moore-Penrose pseudoinverse
    res_float = np.linalg.pinv(coefs).dot(values)
    # Filter negative values
    res = np.where(res_float > 0, res_float, 0)
    # Round values
    res = np.round(res).astype(int)
    res = np.where(res <= n_images, res, 0)
    err = res - res_float
    num_mod = np.sum(res) - n_images
    if num_mod > 0:
        inds = np.argsort(-err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] -= 1
    elif num_mod < 0:
        inds = np.argsort(err)
        mod_inds = inds[:num_mod]
        mod_res = res.copy()
        mod_res[mod_inds] += 1
    else:
        mod_res = res

    return res, mod_res

def get_irlg_res(cls_rec_probs, b_grad, gt_label, num_classes, num_images, simplified=False):
    labels, existences, num_instances, num_instances_nonzero = get_label_stats(gt_label, num_classes)
    # Recovered Labels
    rec_instances, mod_rec_instances = sim_iLRG(cls_rec_probs, b_grad, labels, num_images) if simplified else iLRG(
        cls_rec_probs,
        b_grad,
        num_classes,
        num_images)
    rec_labels = labels if simplified else list(np.where(rec_instances > 0)[0])
    rec_instances_nonzero = rec_instances if simplified else rec_instances[rec_labels]
    rec_existences = [1 if i in rec_labels else 0 for i in range(num_classes)]
    # Calculate Class-wise Acc, Instance-wise Acc and Recall
    leacc = 1.0 if simplified else accuracy_score(existences, rec_existences)
    lnacc = accuracy_score(num_instances_nonzero if simplified else num_instances, list(rec_instances))
    irec = sum([rec_instances[i] if rec_instances[i] <= num_instances_nonzero[i] else num_instances_nonzero[i] for i in
                range(len(labels))]) / num_images if simplified else sum(
        [rec_instances[i] if rec_instances[i] <= num_instances[i] else num_instances[i] for i in labels]) / num_images
    # Print results
    print('Ground-truth Labels: ' + ','.join(str(l) for l in labels))
    print('Ground-truth Num of Instances: ' + ','.join(str(num_instances[l]) for l in labels))
    print('Our Recovered Labels: ' + ','.join(str(l) for l in rec_labels) + ' | LeAcc: %.3f' % leacc)
    prefix = 'Our Recovered Num of Instances by Simplified Method: ' if simplified else 'Our Recovered Num of Instances: '
    print(prefix + ','.join(str(l) for l in list(rec_instances_nonzero)) +
               ' | LnAcc: %.3f | IRec: %.3f' % (
                   lnacc, irec))
    res = [rec_labels, rec_instances_nonzero, rec_instances, existences, mod_rec_instances]
    metrics = [leacc, lnacc, irec]
    return res, metrics

def get_target_data(dataset, labels, start_id=0, device='cpu'):
    images = []
    target_id = start_id
    for i in range(len(labels)):
        while True:
            image, label = dataset[target_id]
            if label == labels[i]:
                images.append(image.float().to(device))
                break
            target_id += 1
            target_id = target_id % len(dataset)
    images = torch.stack(images)
    return images, target_id


def get_data(dataset,
             num_images,
             num_classes,
             start_id=0,
             num_uniform_cls=5,
             num_target_cls=5,
             data_distribution='random',
             device='cpu'):
    images, labels = [], []
    if data_distribution == 'extreme':
        cnt = 0
        target_id = start_id
        extreme_class = random.randint(0, num_classes - 1)
        while cnt < num_images:
            image, label = dataset[target_id]
            if label == extreme_class:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cnt += 1
            target_id += 1
            target_id = target_id % len(dataset)
    elif data_distribution == 'random':
        idx_list = random.sample(range(len(dataset)), num_images)
        for idx in idx_list:
            image, label = dataset[idx]
            images.append(image.float().to(device))
            labels.append(torch.as_tensor((label,), device=device))
        target_id = idx_list[0]
    elif data_distribution == 'balanced':
        target_id = start_id
        uniform_clses = random.sample(range(num_classes), num_uniform_cls)
        num_per_cls = num_images // num_uniform_cls
        cls_cnt = {cls: 0 for cls in uniform_clses}
        while min(list(cls_cnt.values())) < num_per_cls:
            image, label = dataset[target_id]
            if label in uniform_clses and cls_cnt[label] < num_per_cls:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)
    elif data_distribution == 'random2':
        target_id = start_id
        target_clses = random.sample(range(num_classes), num_target_cls)
        # target_clses = random.sample([8, 9, 10, 11, 16, 20, 22, 28, 34, 40, 45, 53, 57, 58, 82, 85, 86, 98],
        #                              num_target_cls)
        # target_clses = random.sample([10, 20, 28, 31, 35, 40, 58, 61, 69, 82, 98, 0, 24],
        #                              num_target_cls)
        # target_clses = random.sample([20, 58, 40, 61, 10, 76, 24, 77, 7, 19, 25, 5, 75],
        #                              num_target_cls)
        random_num = split_integer(num_images, num_target_cls)
        cls_num = {target_clses[i]: random_num[i] for i in range(num_target_cls)}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'custom_imbalanced':
        target_id = start_id
        target_clses = [0, 18, 92]
        cls_num = {0: 1, 18: num_images - 2, 92: 1}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'custom_balanced':
        target_id = start_id
        target_clses = [0, 18, 92]
        cls_num = {0: num_images // 3, 18: num_images // 3, 92: num_images // 3}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'sim':
        target_id = start_id
        target_clses = random.sample(range(num_classes), num_target_cls)
        random_num = split_integer(num_images, num_target_cls)
        cls_num = {target_clses[i]: random_num[i] for i in range(num_target_cls)}
        cls_cnt = {cls: 0 for cls in target_clses}
        while sum(list(cls_cnt.values())) < num_images:
            image, label = dataset[target_id]
            if label in target_clses and cls_cnt[label] < cls_num[label]:
                for _ in range(cls_num[label]):
                    images.append(image.float().to(device))
                    labels.append(torch.as_tensor((label,), device=device))
                    cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)

    elif data_distribution == 'unique':
        target_id = start_id
        uniform_clses = random.sample(range(num_classes), num_images)
        num_per_cls = 1
        cls_cnt = {cls: 0 for cls in uniform_clses}
        while min(list(cls_cnt.values())) < num_per_cls:
            image, label = dataset[target_id]
            if label in uniform_clses and cls_cnt[label] < num_per_cls:
                images.append(image.float().to(device))
                labels.append(torch.as_tensor((label,), device=device))
                cls_cnt[label] += 1
            target_id += 1
            target_id = target_id % len(dataset)
    images = torch.stack(images)
    labels = torch.cat(labels)
    return images, labels, target_id


def get_grads(outs, labels, model, loss_fn, rec=False):
    loss = loss_fn(outs, labels)
    model.zero_grad()
    grads = torch.autograd.grad(loss, model.parameters()) if rec else torch.autograd.grad(loss, model.fc.parameters())
    grads = list((_.detach().cpu().clone() for _ in grads))
    return grads