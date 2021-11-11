"""
Created on Wed Jan 23 10:15:27 2019

@author: aamir-mustafa
Implementation Part 2 of Paper:
    "Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks"

Here it is not necessary to save the best performing model (in terms of accuracy). The model with high robustness
against adversarial attacks is chosen.

"""

import argparse
import datetime
# Essential Imports
import os
import sys
sys.path.append("/home1/machen/meta_perturbations_black_box_attack")
import time

import glog as log
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from adversarial_defense.prototype_conformity_loss.contrastive_proximity import Con_Proximity
from adversarial_defense.prototype_conformity_loss.proximity import Proximity
from adversarial_defense.prototype_conformity_loss.utils import AverageMeter
from config import PY_ROOT, CLASS_NUM
from dataset.dataset_loader_maker import DataLoaderMaker
from adversarial_defense.model.feature_defense_model import FeatureDefenseModel


def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


parser = argparse.ArgumentParser("Prototype Conformity Loss Implementation")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--schedule', type=int, nargs='+', default=[142, 230, 360],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--lr_model', type=float, default=0.01, help="learning rate for CE Loss")
parser.add_argument('--lr_prox', type=float, default=0.5, help="learning rate for Proximity Loss")  # as per paper
parser.add_argument('--weight-prox', type=float, default=1, help="weight for Proximity Loss")  # as per paper
parser.add_argument('--lr_conprox', type=float, default=0.0001,
                    help="learning rate for Con-Proximity Loss")  # as per paper
parser.add_argument('--weight-conprox', type=float, default=0.0001,
                    help="weight for Con-Proximity Loss")  # as per paper
parser.add_argument('--max-epoch', type=int, default=400)
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay")
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--gpu', required=True, type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', type=str, default="pcl_resnet-50")
parser.add_argument("--dataset",type=str, required=True)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    train_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.train_batch, True)
    test_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.test_batch, False)
    model_path = '{}/train_pytorch_model/adversarial_train/pl_loss/pcl_train_{}@{}.pth.tar'.format(
        PY_ROOT, args.dataset, args.arch)
    set_log_file(os.path.dirname(model_path) + "/train_{}_{}.log".format(args.dataset, args.arch))

    if use_gpu:
        log.info("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args(args)
    log.info("After trained over, the model will be saved to {}".format(model_path))
    # Loading the Model
    model = FeatureDefenseModel(args.dataset, args.arch, no_grad=False)
    model = model.cuda()
    num_classes = CLASS_NUM[args.dataset]

    criterion_xent = nn.CrossEntropyLoss()
    criterion_prox_1024 = Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_prox_256 = Proximity(num_classes=num_classes, feat_dim=256, use_gpu=use_gpu)

    criterion_conprox_1024 = Con_Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_conprox_256 = Con_Proximity(num_classes=num_classes, feat_dim=256, use_gpu=use_gpu)

    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=1e-04, momentum=0.9)

    optimizer_prox_1024 = torch.optim.SGD(criterion_prox_1024.parameters(), lr=args.lr_prox)
    optimizer_prox_256 = torch.optim.SGD(criterion_prox_256.parameters(), lr=args.lr_prox)

    optimizer_conprox_1024 = torch.optim.SGD(criterion_conprox_1024.parameters(), lr=args.lr_conprox)
    optimizer_conprox_256 = torch.optim.SGD(criterion_conprox_256.parameters(), lr=args.lr_conprox)

    start_time = time.time()
    resume_epoch = 0

    softmax_model_path = '{}/train_pytorch_model/adversarial_train/pl_loss/benign_image_{}@{}.pth.tar'.format(
        PY_ROOT, args.dataset, args.arch)
    assert os.path.exists(softmax_model_path), "{} does not exist!".format(softmax_model_path)
    state_dict = torch.load(softmax_model_path, map_location=lambda storage, location: storage)
    model.cnn.load_state_dict(state_dict["state_dict"])
    log.info("Load softmax pretrained model from {} done".format(softmax_model_path))
    optimizer_model.load_state_dict(state_dict["optimizer"])


    # if os.path.exists(model_path):
    #     state_dict = torch.load(model_path, map_location=lambda storage, location: storage)
    #     resume_epoch = state_dict["epoch"]
    #     model.load_state_dict(state_dict["state_dict"])
    #     optimizer_model.load_state_dict(state_dict["optimizer_model"])
    #     optimizer_prox_1024.load_state_dict(state_dict["optimizer_prox_1024"])
    #     optimizer_prox_256.load_state_dict(state_dict["optimizer_prox_256"])
    #     optimizer_conprox_1024.load_state_dict(state_dict["optimizer_conprox_1024"])
    #     optimizer_conprox_256.load_state_dict(state_dict["optimizer_conprox_256"])

    for epoch in range(resume_epoch, args.max_epoch):

        adjust_learning_rate(optimizer_model, epoch)
        adjust_learning_rate_prox(optimizer_prox_1024, epoch)
        adjust_learning_rate_prox(optimizer_prox_256, epoch)

        adjust_learning_rate_conprox(optimizer_conprox_1024, epoch)
        adjust_learning_rate_conprox(optimizer_conprox_256, epoch)

        log.info("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        train(model, criterion_xent, criterion_prox_1024, criterion_prox_256,
              criterion_conprox_1024, criterion_conprox_256,
              optimizer_model, optimizer_prox_1024, optimizer_prox_256,
              optimizer_conprox_1024, optimizer_conprox_256,
              train_loader, use_gpu, num_classes, epoch)

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            log.info("==> Test")  # Tests after every 10 epochs
            acc, err = test(model, test_loader)
            log.info("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
        state_ = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                  'optimizer_model': optimizer_model.state_dict(),
                  'optimizer_prox_1024': optimizer_prox_1024.state_dict(),
                  'optimizer_prox_256': optimizer_prox_256.state_dict(),
                  'optimizer_conprox_1024': optimizer_conprox_1024.state_dict(),
                  'optimizer_conprox_256': optimizer_conprox_256.state_dict(), }
        torch.save(state_, model_path)
        log.info("Epoch {} over".format(epoch))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    log.info("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_prox_1024, criterion_prox_256,
          criterion_conprox_1024, criterion_conprox_256,
          optimizer_model, optimizer_prox_1024, optimizer_prox_256,
          optimizer_conprox_1024, optimizer_conprox_256,
          trainloader, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()  # Computes and stores the average and current value
    prox_losses_1024 = AverageMeter()
    prox_losses_256 = AverageMeter()

    conprox_losses_1024 = AverageMeter()
    conprox_losses_256 = AverageMeter()
    losses = AverageMeter()

    # Batchwise training
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        feats128, feats256, feats1024, outputs = model(data)
        loss_xent = criterion_xent(outputs, labels)

        loss_prox_1024 = criterion_prox_1024(feats1024, labels)
        loss_prox_256 = criterion_prox_256(feats256, labels)

        loss_conprox_1024 = criterion_conprox_1024(feats1024, labels)
        loss_conprox_256 = criterion_conprox_256(feats256, labels)

        loss_prox_1024 *= args.weight_prox
        loss_prox_256 *= args.weight_prox

        loss_conprox_1024 *= args.weight_conprox
        loss_conprox_256 *= args.weight_conprox

        loss = loss_xent + loss_prox_1024 + loss_prox_256 - loss_conprox_1024 - loss_conprox_256  # total loss
        optimizer_model.zero_grad()

        optimizer_prox_1024.zero_grad()
        optimizer_prox_256.zero_grad()

        optimizer_conprox_1024.zero_grad()
        optimizer_conprox_256.zero_grad()

        loss.backward()
        optimizer_model.step()

        for param in criterion_prox_1024.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_1024.step()

        for param in criterion_prox_256.parameters():
            param.grad.data *= (1. / args.weight_prox)
        optimizer_prox_256.step()

        for param in criterion_conprox_1024.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_1024.step()

        for param in criterion_conprox_256.parameters():
            param.grad.data *= (1. / args.weight_conprox)
        optimizer_conprox_256.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        prox_losses_1024.update(loss_prox_1024.item(), labels.size(0))
        prox_losses_256.update(loss_prox_256.item(), labels.size(0))

        conprox_losses_1024.update(loss_conprox_1024.item(), labels.size(0))
        conprox_losses_256.update(loss_conprox_256.item(), labels.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            log.info("Batch {}/{}\t Loss {:.6f} ({:.6f})  XentLoss {:.6f} ({:.6f})  ProxLoss_1024 {:.6f} ({:.6f}) ProxLoss_256 {:.6f} ({:.6f}) \n ConProxLoss_1024 {:.6f} ({:.6f}) ConProxLoss_256 {:.6f} ({:.6f}) " \
                .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                        prox_losses_1024.val, prox_losses_1024.avg, prox_losses_256.val, prox_losses_256.avg,
                        conprox_losses_1024.val, conprox_losses_1024.avg, conprox_losses_256.val,
                        conprox_losses_256.avg))


def test(model, testloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if True:
                data, labels = data.cuda(), labels.cuda()
            feats128, feats256, feats1024, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).float().sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_model'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_model'] = state['lr_model']


def adjust_learning_rate_prox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_prox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_prox'] = state['lr_prox']


def adjust_learning_rate_conprox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_conprox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_conprox'] = state['lr_conprox']


if __name__ == '__main__':
    main()





