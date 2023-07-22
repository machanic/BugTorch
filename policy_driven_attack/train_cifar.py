import sys
import os
import torchvision.transforms as transforms

#from torchvision import transforms
from PIL import Image
sys.path.append(os.getcwd())
from config import PROJECT_PATH
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from config import IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
import torch.nn.functional as F
from torch.optim import SGD

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', type=str, default="conv3")
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument("--dataset", type=str, choices=list(IN_CHANNELS.keys()), required=True,help="CIFAR-10")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate_accuracy', dest='evaluate_accuracy', action='store_true',
                    help='evaluate_accuracy model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')


def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    # main_train_worker(args)
    #val_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, True)
    print("2")
    val_loader =DataLoaderMaker.get_imgid_img_label_data_loader(args.dataset, args.batch_size, True, False)
    image_classifier_loss = nn.CrossEntropyLoss().cuda()
    network = StandardModel(args.dataset, args.arch, no_grad=False, load_pretrained=True)
    network.cuda()
    network.eval()
    validate(val_loader, network, image_classifier_loss, args)


def main_train_worker(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    print("=> creating model '{}'".format(args.arch))
    network = StandardModel(args.dataset, args.arch, no_grad=False,load_pretrained=False)
    model_path = '{}/train_pytorch_model/real_image_model/{}@{}@epoch_{}@lr_{}@batch_{}.pth.tar'.format(
       PROJECT_PATH, args.dataset, args.arch, args.epochs, args.lr, args.batch_size)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("after train, model will be saved to {}".format(model_path))
    network.cuda()
    image_classifier_loss = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(network.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True
    train_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, True)
    val_loader = DataLoaderMaker.get_img_label_data_loader(args.dataset, args.batch_size, False)

    for epoch in range(0, args.epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        # train_simulate_grad_mode for one epoch
        train(train_loader, network, image_classifier_loss, optimizer, epoch, args)
        # evaluate_accuracy on validation set
        validate(val_loader, network, image_classifier_loss, args)
        # remember best acc@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=model_path)



def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train_simulate_grad_mode mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1,  = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    import torch
    import pretrainedmodels.utils as utils
    from PIL import Image

    # switch to evaluate_accuracy mode
    model.eval()
    print(model.input_size)
    with torch.no_grad():
        end = time.time()
        for i, (_, input, target) in enumerate(val_loader):
            if args.dataset == "ImageNet" and model.input_size[-1] != 299:
                #print(1)
                input = F.interpolate(input, size=(model.input_size[-2], model.input_size[-1]),
                                      mode='bicubic', align_corners=False)# nearest | linear | bilinear | bicubic | trilinear

                # transformations depending on the model
                # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
            '''
            if input.shape[0] == 1:
                tensor = input.squeeze(0)
                # 将 RGB 通道顺序转换为 BGR 通道顺序
            bgr_tensor = tensor[[2, 1, 0], :, :]
            bgr_tensor = bgr_tensor.unsqueeze(0)
            input = bgr_tensor
            '''

            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1,  = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            # top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
        print('Validate Set Acc@1 {top1.avg:.3f}'
              .format(top1=top1))
    return top1.avg


class ToBGR(object):
    def __init__(self, tensor):
        # 将 RGB 通道顺序转换为 BGR 通道顺序
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            # 将 RGB 通道顺序转换为 BGR 通道顺序
        bgr_tensor = tensor[[2, 1, 0], :, :]
        bgr_tensor = bgr_tensor.unsqueeze(0)
        self.bgr_tensor = bgr_tensor.cuda()
    def getBGR(self):
        return self.bgr_tensor

def save_checkpoint(state, filename='traditional_dl.pth.tar'):
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
