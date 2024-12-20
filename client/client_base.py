import torch.nn as nn
import torch.optim as optim
from utils import accuracy, average_weights, sum_list, global_acc
from sklearn.metrics import accuracy_score
import gc
from models import get_model
from utils import average_weights,global_acc,AverageMeter
from llg import get_label_stats,get_emb,post_process_emb,get_irlg_res
import torch

class Client(object):
    def __init__(self, args, Loader_train, idx, device, model_name='resnet18'):
        self.args = args
        self.trainloader = Loader_train
        self.idx = idx
        self.device = device
        channel = 3
        self.model = get_model(model_name=model_name,
                               net_params=(args.n_classes, channel, self.args.hidden),
                               device=device,
                               n_hidden=1,
                               n_dim=300,
                               batchnorm=False,
                               dropout=True,
                               tanh=False,
                               leaky_relu=False).cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)

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

    def iRLG(self):
        self.model.train()

        average_acc = 0
        average_irec = 0
        average_Leacc = 0

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, self.args.n_classes)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()

            grads = torch.autograd.grad(loss, self.model.fc.parameters())
            grads = list((_.detach().cpu().clone() for _ in grads))
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.max(probs, 1)[1].cpu()

            w_grad, b_grad = grads[-2], grads[-1]

            cls_rec_probs = []

            for i in range(self.args.n_classes):
                cls_rec_emb = get_emb(w_grad[i], b_grad[i])
                cls_rec_prob = post_process_emb(embedding=cls_rec_emb,
                                                model=self.model,
                                                device=self.device,
                                                alpha=1)
                cls_rec_probs.append(cls_rec_prob)

            res, metrics = get_irlg_res(cls_rec_probs=cls_rec_probs,
                                        b_grad=b_grad,
                                        gt_label=targets,
                                        num_classes=self.args.n_classes,
                                        num_images=self.args.batch_size,
                                        simplified=False)
            print(num_instances)
            average_acc += metrics[1]
            average_irec += metrics[2]
            average_Leacc += metrics[0]

        average_Leacc = average_Leacc / len(self.trainloader)
        average_acc = average_acc / len(self.trainloader)
        average_irec = average_irec / len(self.trainloader)
        print('average acc:', average_acc)
        print('average irec:', average_irec)
        return average_Leacc, average_irec

    # def RLU(self):
    #     self.model.train()
    #
    #     average_acc = 0
    #     average_irec = 0
    #     average_cAcc = 0
    #
    #     predictions_softmax = estimate_static_RLU(self.model)
    #
    #     for batch_idx, (inputs, targets) in enumerate(self.trainloader):
    #         # measure data loading time
    #         labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, args.n_classes)
    #
    #         inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    #         inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    #
    #         # compute output
    #         outputs, _ = self.model(inputs)
    #         loss = self.criterion(outputs, targets)
    #
    #         grads = torch.autograd.grad(loss, self.model.fc.parameters())
    #         grads = list((_.detach().cpu().clone() for _ in grads))
    #         probs = torch.softmax(outputs, dim=-1)
    #         preds = torch.max(probs, 1)[1].cpu()
    #
    #         w_grad, b_grad = grads[-2], grads[-1]
    #         n = estimated_entropy_from_grad(predictions_softmax, b_grad.detach().cpu().tolist())
    #         class_existences = [1 if n[i] > 0 else 0 for i in range(len(n))]
    #
    #         cAcc = accuracy_score(existences, class_existences)
    #         acc = accuracy_score(num_instances, n)
    #         res = np.where(n < num_instances, n, num_instances)
    #         irec = sum(
    #             [n[i] if n[i] <= num_instances[i] else num_instances[i] for i in labels]) / args.batch_size
    #         print(num_instances)
    #         print(n)
    #         print('acc:', acc)
    #         print('irec:', irec)
    #         average_acc += acc
    #         average_irec += irec
    #         average_cAcc += cAcc
    #
    #     average_acc = average_acc / len(self.trainloader)
    #     average_irec = average_irec / len(self.trainloader)
    #     average_cAcc = average_cAcc / len(self.trainloader)
    #
    #     print('average acc:', average_acc)
    #     print('average irec:', average_irec)
    #     return average_cAcc, average_irec
    #
    # def LLG(self):
    #     self.model.train()
    #
    #     average_acc = 0
    #     average_irec = 0
    #     average_cAcc = 0
    #
    #     for batch_idx, (inputs, targets) in enumerate(self.trainloader):
    #         # measure data loading time
    #         labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, args.n_classes)
    #
    #         inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    #         inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    #
    #         # compute output
    #         outputs, _ = self.model(inputs)
    #         loss = self.criterion(outputs, targets)
    #
    #         grads = torch.autograd.grad(loss, self.model.fc.parameters())
    #         grads = list((_.detach().cpu().clone() for _ in grads))
    #         probs = torch.softmax(outputs, dim=-1)
    #         preds = torch.max(probs, 1)[1].cpu()
    #
    #         w_grad, b_grad = grads[-2], grads[-1]
    #
    #         h1_extraction = []
    #         impact_acc = 0
    #
    #         gradients_for_prediction = torch.sum(w_grad, dim=-1)
    #         # filter negative values
    #         for i_cg, class_gradient in enumerate(gradients_for_prediction):
    #             if class_gradient < 0:
    #                 h1_extraction.append((i_cg, class_gradient))
    #                 impact_acc += class_gradient.item()
    #
    #         impact = (impact_acc / args.batch_size) * (1 + 1 / args.n_classes)
    #
    #         prediction = []
    #
    #         for (i_c, _) in h1_extraction:
    #             prediction.append(i_c)
    #             gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)
    #
    #         for _ in range(args.batch_size - len(prediction)):
    #             # add minimal candidate, likely to be doubled, to prediction
    #             min_id = torch.argmin(gradients_for_prediction).item()
    #             prediction.append(min_id)
    #
    #             # add the mean value of one occurrence to the candidate
    #             gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-impact)
    #
    #         n = []
    #         for i in range(args.n_classes):
    #             n.append(prediction.count(i))
    #
    #         class_existences = [1 if n[i] > 0 else 0 for i in range(len(n))]
    #         cAcc = accuracy_score(existences, class_existences)
    #         acc = accuracy_score(num_instances, n)
    #         res = np.where(n < num_instances, n, num_instances)
    #         irec = sum([n[i] if n[i] <= num_instances[i] else num_instances[i] for i in labels]) / args.batch_size
    #         print(num_instances)
    #         print(n)
    #         print('acc:', acc)
    #         print('irec:', irec)
    #         average_acc += acc
    #         average_irec += irec
    #         average_cAcc += cAcc
    #
    #     average_acc = average_acc / len(self.trainloader)
    #     average_irec = average_irec / len(self.trainloader)
    #     average_cAcc = average_cAcc / len(self.trainloader)
    #
    #     print('average acc:', average_acc)
    #     print('average irec:', average_irec)
    #     return average_cAcc, average_irec
    #
    # def LLGp(self):
    #     self.model.train()
    #
    #     average_acc = 0
    #     average_irec = 0
    #     average_cAcc = 0
    #
    #     impact, offset = estimate_static_LLG(self.model)
    #
    #     for batch_idx, (inputs, targets) in enumerate(self.trainloader):
    #         # measure data loading time
    #         labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, args.n_classes)
    #
    #         inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    #         inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    #
    #         # compute output
    #         outputs, _ = self.model(inputs)
    #         loss = self.criterion(outputs, targets)
    #
    #         grads = torch.autograd.grad(loss, self.model.fc.parameters())
    #         grads = list((_.detach().cpu().clone() for _ in grads))
    #         probs = torch.softmax(outputs, dim=-1)
    #         preds = torch.max(probs, 1)[1].cpu()
    #
    #         w_grad, b_grad = grads[-2], grads[-1]
    #
    #         h1_extraction = []
    #
    #         gradients_for_prediction = torch.sum(w_grad, dim=-1)
    #
    #         # filter negative values
    #         for i_cg, class_gradient in enumerate(gradients_for_prediction):
    #             if class_gradient < 0:
    #                 h1_extraction.append((i_cg, class_gradient))
    #
    #         gradients_for_prediction -= offset
    #
    #         prediction = []
    #
    #         for (i_c, _) in h1_extraction:
    #             prediction.append(i_c)
    #             gradients_for_prediction[i_c] = gradients_for_prediction[i_c].add(-impact)
    #
    #         for _ in range(args.batch_size - len(prediction)):
    #             # add minimal candidate, likely to be doubled, to prediction
    #             min_id = torch.argmin(gradients_for_prediction).item()
    #             prediction.append(min_id)
    #
    #             # add the mean value of one occurrence to the candidate
    #             gradients_for_prediction[min_id] = gradients_for_prediction[min_id].add(-impact)
    #
    #         n = []
    #         for i in range(args.n_classes):
    #             n.append(prediction.count(i))
    #
    #         class_existences = [1 if n[i] > 0 else 0 for i in range(len(n))]
    #         cAcc = accuracy_score(existences, class_existences)
    #         acc = accuracy_score(num_instances, n)
    #         res = np.where(n < num_instances, n, num_instances)
    #         irec = sum([n[i] if n[i] <= num_instances[i] else num_instances[i] for i in labels]) / args.batch_size
    #         print(num_instances)
    #         print(n)
    #         print('acc:', acc)
    #         print('irec:', irec)
    #         average_acc += acc
    #         average_irec += irec
    #         average_cAcc += cAcc
    #
    #     average_acc = average_acc / len(self.trainloader)
    #     average_irec = average_irec / len(self.trainloader)
    #     average_cAcc = average_cAcc / len(self.trainloader)
    #
    #     print('average acc:', average_acc)
    #     print('average irec:', average_irec)
    #     return average_cAcc, average_irec
    #
    # def ZLG(self):
    #     self.model.train()
    #
    #     average_acc = 0
    #     average_irec = 0
    #     average_cAcc = 0
    #
    #     O_bar, pj = estimate_static_ZLG(self.model)
    #     for batch_idx, (inputs, targets) in enumerate(self.trainloader):
    #         # measure data loading time
    #         labels, existences, num_instances, num_instances_nonzero = get_label_stats(targets, args.n_classes)
    #
    #         inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
    #         inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
    #
    #         # compute output
    #         outputs, _ = self.model(inputs)
    #         loss = self.criterion(outputs, targets)
    #
    #         grads = torch.autograd.grad(loss, self.model.fc.parameters())
    #         grads = list((_.detach().cpu().clone() for _ in grads))
    #         probs = torch.softmax(outputs, dim=-1)
    #         preds = torch.max(probs, 1)[1].cpu()
    #
    #         w_grad, b_grad = grads[-2], grads[-1]
    #
    #         gradients_for_prediction = torch.sum(w_grad, dim=-1)
    #
    #         n = np.zeros(args.n_classes)
    #         for i in range(args.n_classes):
    #             nj = pj[i].detach().cpu() - gradients_for_prediction[i] / O_bar.detach().cpu()
    #             n[i] = np.max(nj.item(), 0)
    #         n = n / np.sum(n)
    #         n = n.tolist()
    #         for i in range(args.n_classes):
    #             n[i] = round(args.batch_size * n[i])
    #
    #         class_existences = [1 if n[i] > 0 else 0 for i in range(len(n))]
    #         cAcc = accuracy_score(existences, class_existences)
    #         acc = accuracy_score(num_instances, n)
    #         res = np.where(n < num_instances, n, num_instances)
    #         irec = sum([n[i] if n[i] <= num_instances[i] else num_instances[i] for i in labels]) / args.batch_size
    #         print(num_instances)
    #         print(n)
    #         print('acc:', acc)
    #         print('irec:', irec)
    #         average_acc += acc
    #         average_irec += irec
    #         average_cAcc += cAcc
    #
    #     average_acc = average_acc / len(self.trainloader)
    #     average_irec = average_irec / len(self.trainloader)
    #     average_cAcc = average_cAcc / len(self.trainloader)
    #
    #     print('average acc:', average_acc)
    #     print('average irec:', average_irec)
    #     return average_cAcc, average_irec

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