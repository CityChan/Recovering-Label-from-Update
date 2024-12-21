import torch.nn as nn
import torch.optim as optim
from utils import accuracy, average_weights, sum_list, global_acc
from sklearn.metrics import accuracy_score
import gc
from models import get_model
from utils import average_weights,global_acc,AverageMeter
from llg import get_label_stats,get_emb,post_process_emb,get_irlg_res
import torch
from client.client_utils import estimate_static_RLU, estimated_entropy_from_grad, estimate_static_RLU_with_posterior
from client.client_utils import estimate_static_LLG,estimate_static_ZLG

import numpy as np
import copy
import scipy

class Client(object):
    def __init__(self, args, Loader_train, idx, device, model_name, aux_dataset):
        self.args = args
        self.trainloader = Loader_train
        self.idx = idx
        self.device = device
        self.aux_dataset = aux_dataset
        channel = 3
        tanh = False
        if args.activation == 'tanh':
            tanh = True
        self.model = get_model(model_name=model_name,
                               net_params=(args.n_classes, channel, self.args.hidden),
                               device=device,
                               n_hidden=1,
                               n_dim=300,
                               batchnorm=False,
                               dropout=True,
                               tanh=tanh,
                               leaky_relu=False).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)
        if args.model == 'resnet18':
            self.latent_dim = 512
        if args.model == 'vgg16':
            self.latent_dim = 4096
        if args.model == 'lenet5':
            self.latent_dim = 84

    def train(self, epoch):
        self.model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses.avg, top1.avg

    def iRLG(self,global_weights):
        self.model.train()

        average_acc = 0
        average_irec = 0
        average_Leacc = 0

        count_computed = 0

        count = 0
        w_grad_epochs = torch.zeros([self.args.n_classes, self.latent_dim])
        b_grad_epochs = torch.zeros([self.args.n_classes])

        targets_epochs = []

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, self.args.n_classes)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            targets_epochs.append(targets)

            # compute output
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            grads = []
            for param in self.model.fc.parameters():
                grads.append(param.grad.detach().cpu().clone())

            probs = torch.softmax(outputs, dim=-1)

            w_grad, b_grad = grads[-2], grads[-1]

            w_grad_epochs += w_grad
            b_grad_epochs += b_grad

            count += 1

            if count == self.args.local_epochs:
                self.load_model(global_weights)
                w_grad_epochs = w_grad_epochs / self.args.local_epochs
                b_grad_epochs = b_grad_epochs / self.args.local_epochs
                count = 0
                count_computed += 1
                cls_rec_probs = []

                for i in range(self.args.n_classes):
                    cls_rec_emb = get_emb(w_grad_epochs[i], b_grad_epochs[i])
                    cls_rec_prob = post_process_emb(embedding=cls_rec_emb,
                                                    model=self.model,
                                                    device=self.device,
                                                    alpha=1)
                    cls_rec_probs.append(cls_rec_prob)

                targets_epochs = torch.cat(targets_epochs, dim=0)

                res, metrics = get_irlg_res(cls_rec_probs=cls_rec_probs,
                                            b_grad=b_grad_epochs,
                                            gt_label=targets_epochs,
                                            num_classes=self.args.n_classes,
                                            num_images=self.args.batch_size * self.args.local_epochs,
                                            simplified=False)

                average_acc += metrics[1]
                average_irec += metrics[2]
                average_Leacc += metrics[0]

                w_grad_epochs = torch.zeros([self.args.n_classes, self.latent_dim])
                b_grad_epochs = torch.zeros([self.args.n_classes])
                targets_epochs = []

        average_Leacc = average_Leacc / count_computed
        average_acc = average_acc / count_computed
        average_irec = average_irec / count_computed
        print('average irec:', average_irec)
        return average_Leacc, average_irec

    def RLU(self, global_weights):
        self.model.train()

        average_acc = 0
        average_irec = 0
        average_cAcc = 0

        count_computed = 0

        count = 0
        b_grad_epochs = torch.zeros([self.args.n_classes])
        w_grad_epochs = torch.zeros([self.args.n_classes, self.latent_dim])
        targets_epochs = []
        self.mu, _ = estimate_static_RLU(self.args, copy.deepcopy(self.model), self.aux_dataset)
        self.O = torch.zeros(self.latent_dim)

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            targets_epochs.append(targets)
            # compute output

            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            grads = []

            for param in self.model.fc.parameters():
                grads.append(param.grad.detach().cpu().clone())

            w_grad, b_grad = grads[-2], grads[-1]

            b_grad_epochs += b_grad
            w_grad_epochs += w_grad
            count += 1

            if count == self.args.local_epochs:
                new_mu, new_shift = estimate_static_RLU(self.args, copy.deepcopy(self.model), self.aux_dataset)
                new_shift = scipy.special.softmax(new_mu)

                targets_epochs = torch.cat(targets_epochs, dim=0)
                targets_epochs = targets_epochs.tolist()

                num_instances = np.zeros(self.args.n_classes)
                for k in range(self.args.n_classes):
                    num_instances[k] = targets_epochs.count(k)

                b_grad_epochs = b_grad_epochs / self.args.local_epochs
                w_grad_epochs = w_grad_epochs / self.args.local_epochs
                for d in range(self.latent_dim):
                    self.O[d] = torch.mean(w_grad_epochs[:, d] / b_grad_epochs)
                count = 0
                count_computed += 1

                rho = np.zeros(self.args.local_epochs)
                gamma = self.args.gamma
                for t in range(self.args.local_epochs):
                    rho[t] = (1 - gamma ** (self.args.local_epochs - t)) / (1 - gamma)
                rho_mean = np.sum(rho) / self.args.local_epochs

                n = estimated_entropy_from_grad(self.args, new_shift*new_shift*rho_mean, b_grad_epochs.detach().cpu().tolist(),
                                                self.args.batch_size * self.args.local_epochs)
                new_shift_softmax = estimate_static_RLU_with_posterior(self.args, n, self.mu, new_mu, self.O)
                n = estimated_entropy_from_grad(self.args, new_shift_softmax*rho_mean,b_grad_epochs.detach().cpu().tolist(), self.args.batch_size*self.args.local_epochs)
                class_existences = [1 if n[i] > 0 else 0 for i in range(len(n))]
                existences = [1 if num_instances[i] > 0 else 0 for i in range(len(num_instances))]

                cAcc = accuracy_score(existences, class_existences)
                acc = accuracy_score(num_instances, n)
                res = np.where(n < num_instances, n, num_instances)
                labels = range(self.args.n_classes)
                irec = sum(
                    [n[i] if n[i] <= num_instances[i] else num_instances[i] for i in labels]) / (
                                   self.args.batch_size * self.args.local_epochs)
                print(num_instances)
                print(n)
                print('irec:', irec)
                average_acc += acc
                average_irec += irec
                average_cAcc += cAcc

                b_grad_epochs = torch.zeros([self.args.n_classes])
                targets_epochs = []
                self.load_model(global_weights)

        average_acc = average_acc / count_computed
        average_irec = average_irec / count_computed
        average_cAcc = average_cAcc / count_computed

        print('average irec:', average_irec)
        return average_cAcc, average_irec

    def LLGp(self, global_weights):
        self.model.train()

        average_acc = 0
        average_irec = 0
        average_cAcc = 0

        count_computed = 0

        count = 0

        w_grad_epochs = torch.zeros([self.args.n_classes, self.latent_dim])
        targets_epochs = []

        impact, offset = estimate_static_LLG(self.args, copy.deepcopy(self.model), self.aux_dataset)

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, self.args.n_classes)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            targets_epochs.append(targets)
            # compute output
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            grads = []
            for param in self.model.fc.parameters():
                grads.append(param.grad.detach().cpu().clone())

            w_grad, b_grad = grads[-2], grads[-1]
            w_grad_epochs += w_grad
            count += 1

            if count == self.args.local_epochs:
                new_impact, new_offset = estimate_static_LLG(self.args, copy.deepcopy(self.model), self.aux_dataset)
                new_impact = (new_impact + impact) / 2
                new_offset = (new_offset + offset) / 2
                targets_epochs = torch.cat(targets_epochs, dim=0)
                targets_epochs = targets_epochs.tolist()
                num_instances = np.zeros(self.args.n_classes)
                for k in range(self.args.n_classes):
                    num_instances[k] = targets_epochs.count(k)
                w_grad_epochs = w_grad_epochs / self.args.local_epochs
                count = 0
                count_computed += 1

                h1_extraction = []

                gradients_for_prediction = torch.sum(w_grad_epochs, dim=-1)

                # filter negative values
                for i_cg, class_gradient in enumerate(gradients_for_prediction):
                    if class_gradient < 0:
                        h1_extraction.append((i_cg, class_gradient))

                gradients_for_prediction -= new_offset
                prediction = []

                for (i_c, _) in h1_extraction:
                    prediction.append(i_c)
                    gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)

                for _ in range(self.args.batch_size - len(prediction)):
                    # add minimal candidate, likely to be doubled, to prediction
                    min_id = torch.argmin(gradients_for_prediction).item()
                    prediction.append(min_id)

                    # add the mean value of one occurrence to the candidate
                    gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-new_impact)

                n = []
                for i in range(self.args.n_classes):
                    n.append(self.args.local_epochs * prediction.count(i))

                class_existences = [1 if n[i] > 0 else 0 for i in range(len(n))]
                cAcc = accuracy_score(existences, class_existences)
                acc = accuracy_score(num_instances, n)
                res = np.where(n < num_instances, n, num_instances)
                labels = range(self.args.n_classes)
                irec = sum([n[i] if n[i] <= num_instances[i] else num_instances[i] for i in labels]) / (
                            self.args.batch_size * self.args.local_epochs)
                print(num_instances)
                print(n)
                print('irec:', irec)
                average_acc += acc
                average_irec += irec
                average_cAcc += cAcc

                w_grad_epochs = torch.zeros([self.args.n_classes, self.latent_dim])
                targets_epochs = []
                self.load_model(global_weights)

        average_acc = average_acc / count_computed
        average_irec = average_irec / count_computed
        average_cAcc = average_cAcc / count_computed

        print('average irec:', average_irec)
        return average_cAcc, average_irec

    def ZLGp(self, global_weights):
        self.model.train()

        average_acc = 0
        average_irec = 0
        average_cAcc = 0

        count_computed = 0

        count = 0

        w_grad_epochs = torch.zeros([self.args.n_classes, self.latent_dim])
        targets_epochs = []

        O_bar, pj = estimate_static_ZLG(self.args, copy.deepcopy(self.model), self.aux_dataset)

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, self.args.n_classes)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_epochs.append(targets)
            # compute output
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            self.optimizer.step()

            grads = []
            for param in self.model.fc.parameters():
                grads.append(param.grad.detach().cpu().clone())

            w_grad, b_grad = grads[-2], grads[-1]
            w_grad_epochs += w_grad
            count += 1

            if count == self.args.local_epochs:
                new_O_bar, new_pj = estimate_static_ZLG(self.args,copy.deepcopy(self.model), self.aux_dataset)
                new_O_bar = (new_O_bar + O_bar) / 2
                new_pj = (new_pj + pj) / 2
                targets_epochs = torch.cat(targets_epochs, dim=0)
                targets_epochs = targets_epochs.tolist()
                num_instances = np.zeros(self.args.n_classes)
                for k in range(self.args.n_classes):
                    num_instances[k] = targets_epochs.count(k)

                w_grad_epochs = w_grad_epochs / self.args.local_epochs

                count = 0
                count_computed += 1

                gradients_for_prediction = torch.sum(w_grad_epochs, dim=-1)
                n = []
                for i in range(self.args.n_classes):
                    nj = self.args.batch_size * self.args.local_epochs * (
                                new_pj[i].detach().cpu() - gradients_for_prediction[i] / new_O_bar.detach().cpu())
                    n.append(max(int(nj.item()), 0))

                prop = (self.args.local_epochs * self.args.batch_size) / sum(n)
                for i in range(self.args.n_classes):
                    n[i] = round(n[i] * prop)

                class_existences = [1 if n[i] > 0 else 0 for i in range(len(n))]
                cAcc = accuracy_score(existences, class_existences)
                acc = accuracy_score(num_instances, n)
                res = np.where(n < num_instances, n, num_instances)
                labels = range(self.args.n_classes)
                irec = sum([n[i] if n[i] <= num_instances[i] else num_instances[i] for i in labels]) / (
                            self.args.batch_size * self.args.local_epochs)
                print(num_instances)
                print(n)
                print('irec:', irec)
                average_acc += acc
                average_irec += irec
                average_cAcc += cAcc
                w_grad_epochs = torch.zeros([self.args.n_classes, self.latent_dim])
                targets_epochs = []
                self.load_model(global_weights)

        average_acc = average_acc / count_computed
        average_irec = average_irec / count_computed
        average_cAcc = average_cAcc / count_computed

        print('average irec:', average_irec)
        return average_cAcc, average_irec

    def local_training(self, global_epoch):
        for epoch in range(self.args.local_epochs):
            self.adjust_learning_rate(epoch + global_epoch * self.args.local_epochs)
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
            train_loss, train_acc = self.train(epoch)
        print(f'Client {self.idx} Training Top 1 Acc at global round {global_epoch} : {train_acc}')

    def adjust_learning_rate(self, epoch):
        global state
        if epoch in self.args.schedule:
            state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = state['lr']

    def load_model(self, global_weights):
        self.model.load_state_dict(global_weights)