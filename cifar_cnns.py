"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import os
import time
import argparse
import torchvision
import torch.optim as optim
from torchvision import datasets
from torch.optim import Adam, SGD
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from models import *

from utils.logger import Logger, savefig
from utils.misc import mkdir_p

import sys
sys.path.append('./optim')
from adar import AdaR

# from adabound import AdaBound
# from adabs import AdaBS
# from radam import RAdam
# from adamw import AdamW
# from padam import PAdam
# from adam_win import Adam_Win
# from adan import Adan
# from nero import Nero
# from yogi import Yogi
# from radam import RAdam
# from fromage import Fromage
# from msvag import MSVAG
# from swats import SWATS

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--decay_epoch', default=150, type=int, help='Number of epochs to decay learning rate')
    parser.add_argument('--model', default='resnet', type=str, help='model',
                        choices=['resnet', 'densenet', 'vggnet'])
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset',
                        choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--run', default=0, type=int, help='number of runs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate')
    parser.add_argument('--initial_lr', default=0.001, type=float,
                        help='initial learning rate of AdaLA')
    
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps for var adam')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--partial', default=8,  type=float, help='PAdam partially adaptive parameter')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=1e-8, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--reset', action = 'store_true',
                        help='whether reset optimizer at learning rate decay')
    parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
    return parser


def build_dataset(args):
    print('==> Preparing data..')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./MNIST', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                './MNIST',
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.1307,
                             ),
                            (0.3081,
                             ))])),
            batch_size=args.batchsize,
            shuffle=False)

    if args.dataset == 'cifar10':

        trainset = datasets.CIFAR10(
            root='../data/CIFAR',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batchsize, shuffle=True)

        testset = datasets.CIFAR10(
            root='../data/CIFAR',
            train=False,
            download=False,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batchsize, shuffle=False)

    if args.dataset == 'cifar100':

        trainset = datasets.CIFAR100(
            root='../data/CIFAR',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batchsize, shuffle=True)

        testset = datasets.CIFAR100(
            root='../data/CIFAR',
            train=False,
            download=False,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batchsize, shuffle=False)

    return train_loader, test_loader


def get_ckpt_name(model='resnet', optimizer='sgd', lr=0.1, initial_lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3, eps=1e-8, weight_decay=5e-4,
                  partial=1/8, lower=0.01, upper=100,
                  reset = False, run = 0, weight_decouple = False, rectify = False):
    name = {
        'sgd': 'lr{}-momentum{}-wdecay{}-run{}'.format(lr, momentum,weight_decay, run),
        'adam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'swats': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'amsgrad': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'nero': 'lr{}-run{}'.format(lr, run),
        'adar': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, weight_decay, eps, run),
        # 'adala': 'lr{}-betas{}-{}-lower{}-upper{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, lower, upper, weight_decay, eps, run),
        'radam': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'padam': 'lr{}-betas{}-{}-partial-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2, partial, weight_decay, eps, run),
        'adamw': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'adabelief': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps, weight_decay, run),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}-wdecay{}-run{}'.format(lr, beta1, beta2, final_lr, gamma,weight_decay, run),
        'fromage': 'lr{}-betas{}-{}-wdecay{}-eps{}-run{}'.format(lr, beta1, beta2,weight_decay, eps, run),
        'yogi':'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps,weight_decay, run),
        'adan':'lr{}-betas{}-{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta1, beta2, eps,weight_decay, run),
        'adam_win':'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps,weight_decay, run),
        'msvag': 'lr{}-betas{}-{}-eps{}-wdecay{}-run{}'.format(lr, beta1, beta2, eps,weight_decay, run),
    
    }[optimizer]
    return '{}-{}-{}-reset{}'.format(model, optimizer, name, str(reset))


def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)


def build_model(args, device, ckpt=None):
    if args.dataset == 'mnist':
        nclass = 10
    elif args.dataset == 'cifar10':
        nclass = 10
    elif args.dataset == 'cifar100':
        nclass = 100
        
    print('==> Building model..')
    net = {
        'resnet': resnet34,
        'densenet': densenet121,
        'vggnet':vggnet11,
    }[args.model](num_classes=nclass)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net


def create_optimizer(args, model_params):
    args.optim = args.optim.lower()
    if args.optim == 'adar':
        return AdaR(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'swats':
    #     return SWATS(model_params, args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'adabs':
    #     return AdaBS(model_params, args.lr, betas=(args.beta1, args.beta2),
    #                  final_lr=args.final_lr, lower=args.lower, upper=args.upper,
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'padam':
    #     return PAdam(model_params, args.lr, partial=1/args.partial, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'radam':
    #     return RAdam(model_params, args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'adabelief':
    #     return AdaBelief(model_params, args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'adabound':
    #     return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
    #                     final_lr=args.final_lr, gamma=args.gamma,
    #                     weight_decay=args.weight_decay)
    # elif args.optim == 'amsgrad':
    #     return torch.optim.Adam(model_params, lr=args.lr, amsgrad=True, betas=(args.beta1,args.beta2),
    #                             weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'nero':
    #     return Nero(model_params,lr=args.lr)
    # elif args.optim == 'adamw':
    #     return AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'msvag':
    #     return MSVAG(model_params,lr=args.lr)
    # elif args.optim == 'fromage':
    #     return Fromage(model_params,lr=args.lr)
    # elif args.optim == 'yogi':
    #     return Yogi(model_params, args.lr, betas=(args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay)
    # elif args.optim == 'adan':
    #     return Adan(model_params,lr=args.lr, betas=(args.beta1,args.beta1, args.beta2),
    #                       weight_decay=args.weight_decay, eps=args.eps)
    # elif args.optim == 'adam_win':
    #     return Adam_Win(model_params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), acceleration_mode='win')
    else:
        print('Optimizer not found')

def train(net, epoch, device, data_loader, optimizer, criterion, args):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    train_loss /= len(data_loader.dataset)
    print('train acc %.3f' % accuracy)

    return train_loss, accuracy


def test(net, device, data_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    test_loss /= len(data_loader.dataset)
    print(' test acc %.3f' % accuracy)

    return test_loss, accuracy

def adjust_learning_rate(optimizer, epoch, step_size=150, gamma=0.1, reset = False):
    for param_group in optimizer.param_groups:
        if epoch % step_size==0 and epoch>0:
            param_group['lr'] *= gamma
        if epoch % 75==0 and epoch>0:
            param_group['lr'] *= gamma

    if  epoch % step_size==0 and epoch>0 and reset:
        optimizer.reset()

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    train_loader, test_loader = build_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              initial_lr=args.initial_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma,
                              eps = args.eps, partial= 1/args.partial,
                              reset=args.reset, run=args.run,
                              weight_decay = args.weight_decay)
    print('ckpt_name')
    

    log_dir=f'logs/res/{args.dataset}-{args.model}-{args.optim}-{args.lr}-decay{args.weight_decay}-seed{args.seed}'
    # if args.optim == 'adabound':
    #     log_dir=f'logs/res/{args.dataset}-{args.model}-{args.optim}-{args.lr}-decay{args.weight_decay}-seed{args.seed}'
    
    # if args.optim == 'adabs':
    #     log_dir=f'logs/res/{args.dataset}-{args.model}-{args.optim}-{args.lr}-final_lr{args.final_lr}-low{args.lower}-up{args.upper}-decay{args.weight_decay}-seed{args.seed}'
    # elif args.optim == 'padam':
    #     log_dir=f'logs/res/{args.dataset}-{args.model}-{args.optim}-{args.lr}-partial{1/args.partial}-decay{args.weight_decay}-seed{args.seed}'
    # elif args.optim == 'adabound':
    #     log_dir=f'logs/res/{args.dataset}-{args.model}-{args.optim}-{args.lr}-final_lr{args.final_lr}-decay{args.weight_decay}-seed{args.seed}'
    # else :
    #     log_dir=f'logs/res/{args.dataset}-{args.model}-{args.optim}-{args.lr}-decay{args.weight_decay}-seed{args.seed}'
        
    log_dir += "-m" + str(args.momentum)
    log_dir += "-epochs" + str(args.epochs)

    title = 'CIFAR-' + args.dataset + args.model + str(args.epochs)

    if not os.path.isdir(log_dir):
        mkdir_p(log_dir)
    logger = Logger(os.path.join(log_dir, 'log.txt'), title=title)
    logger.set_names(['Train Loss,', 'Train Acc,', 'Valid Loss,', 'Val Acc,', 'Time'])

    if args.resume:
        ckpt = load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']

        curve = os.path.join('curve', ckpt_name)     
        curve = torch.load(curve)
        train_accuracies = curve['train_acc']
        test_accuracies = curve['test_acc']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1
        train_accuracies = []
        test_accuracies = []

    net = build_model(args, device, ckpt=ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, net.parameters())    

    for epoch in range(start_epoch + 1, args.epochs):
        start = time.time()
        #scheduler.step()
        adjust_learning_rate(optimizer, epoch, step_size=args.decay_epoch, gamma=args.lr_gamma, reset = args.reset)
        train_loss, train_acc = train(net, epoch, device, train_loader, optimizer, criterion, args)
        test_loss, test_acc = test(net, device, test_loader, criterion)
        end = time.time()
        print('Time: {}'.format(end-start))

        print(args.optim)
        
        logger.append([train_loss, train_acc, test_loss, test_acc, (end - start)])
        

        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        if not os.path.isdir('curve'):
            os.mkdir('curve')
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))

    logger.close()

if __name__ == '__main__':
    main()
