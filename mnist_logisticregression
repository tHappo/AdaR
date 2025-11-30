import os
import time
import math
import pickle
import numpy as np
import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer as optim

from torchvision import datasets, transforms

import sys
sys.path.append('./optim/')
# from adar import AdaR
# from adabound import AdaBound
# from adashift import AdaShift
from adaplus import AdaPlus
from adam_win import Adam_Win
from nosadam import NosAdam

from utils.misc import mkdir_p
from utils.logger import Logger, savefig

tqdm = lambda x: x

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--optim', default='sgd', type=str)
parser.add_argument('--model', default='lr',
                    help='model names (default: logistic regression)')

parser.add_argument('--num_epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--eps', default=1e-8, type=float, metavar='E',
                    help='eps for adaptive methods')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--final-lr', default=0.01, type=float, metavar='F',
                    help='final step size for AdaBS and AdaBound')
parser.add_argument('--lower', default=0.02, type=float, metavar='L',
                    help='lower-step-size bound for AdaBS ')
parser.add_argument('--upper', default=50, type=float, metavar='U',
                    help='upper-step-size bound for AdaBS')
parser.add_argument('--partial', default=4, type=float, metavar='P',
                    help='partial term for PAdam')

parser.add_argument('--seed', default=1234, type=int,
                    help='seed for initializing training. ')

args = parser.parse_args()

trainset = datasets.MNIST('../data/MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

testset = datasets.MNIST('../data/MNIST', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

def getLR(optim):
    for param_group in optimizer.param_groups:
        print(f"lr is {param_group['lr']}")

class MLP(nn.Module):
    def __init__(self, depth, width, bias):
        super(MLP, self).__init__()
        
        self.initial = nn.Linear(784, width, bias=bias)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=bias) for _ in range(depth-2)])
        self.final = nn.Linear(width, 10, bias=bias)
        
    def forward(self, x):
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)
        return self.final(x)

def train_network(epoch, model, optimizer, loss_fn, lr_scheduler):

    model.train()
   
    correct = 0
    train_loss = 0
    total = len(train_loader.dataset)

    for data, target in tqdm(train_loader):
        data, target = (data.cuda(), target.cuda())

        optimizer.zero_grad()

        y_pred = model(data.reshape(-1, 28*28))
        loss = loss_fn(y_pred, target)
        train_loss += loss
        correct += (target == y_pred.max(dim=1)[1]).sum().item()

        model.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step()
    train_loss /= total
    print(f"\nEpoch {epoch} train_acc {100. * correct/total} lr is {optimizer.param_groups[0]['lr']}")
    
    return train_loss, (correct/total)


def test(model, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = (data.cuda(), target.cuda())
            y_pred = model(data.reshape(-1, 28*28))
            test_loss += loss_fn(y_pred, target) # sum up batch loss
            y_pred = y_pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += y_pred.eq(target.view_as(y_pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    return test_loss, acc

def main():

    if args.optim == 'adar':
        log_dir=f'logs/res/{args.model}-{args.optim}-{args.lr}-decay{args.weight_decay}-eps{args.eps}-seed{args.seed}'
        log_dir += "-epochs" + str(args.num_epochs)
        
    elif args.optim == 'sgd':
        log_dir=f'logs/res/{args.model}-{args.optim}-{args.lr}-decay{args.weight_decay}-seed{args.seed}'
        log_dir += "-m" + str(args.momentum)
        log_dir += "-epochs" + str(args.num_epochs)

    elif args.optim == 'adabound':
        log_dir=f'logs/res/{args.model}-{args.optim}-{args.lr}-final_lr{args.final_lr}-decay{args.weight_decay}-seed{args.seed}'
        log_dir += "-epochs" + str(args.num_epochs)
        
    else :
        log_dir=f'logs/res/{args.model}-{args.optim}-{args.lr}-decay{args.weight_decay}-seed{args.seed}'
        log_dir += "-epochs" + str(args.num_epochs)

    title = 'MNIST-' + args.model
    if not os.path.isdir(log_dir):
        mkdir_p(log_dir)
    logger = Logger(os.path.join(log_dir, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    if args.model == 'lr':
        model = nn.Linear(784, 10).cuda()
    elif args.model == 'mlp':
        model = MLP(3, 784, True).cuda()

    """
    width = 784
    depth = 100
    bias = True
    """

    loss_fn = nn.CrossEntropyLoss()
    
    if args.optim == "adar":
        optimizer = AdaR(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    elif args.optim == "adam_win":
        optimizer = Adam_Win(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, acceleration_mode='win')
    elif args.optim == "nosadam":
        optimizer = NosAdam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    elif args.optim == "adaplus":
        optimizer = AdaPlus(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
    elif args.optim == "adashift":
        optimizer = AdaShift(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    elif args.optim == "adabound":
        optimizer = AdaBound(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, final_lr=args.final_lr)
   
      
    lr_lambda = lambda x: 0.9**x
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("Running MNIST Experiment")
    print("\n==========================================================")
    print(f"optimizer {type(optimizer).__name__}, initial learning rate {args.lr}, seed {args.seed}\n")
    
    best_acc = 0

    for epoch in range(1, args.num_epochs + 1):
        start = time.time()
    
        train_loss, train_acc = train_network(epoch, model, optimizer, loss_fn, lr_scheduler)
        test_loss, test_acc = test(model, loss_fn)

        end = time.time()
        
        print( f'Final test acc: {100. * test_acc}' + '  Time: {}'.format(end-start))
        logger.append([optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc])


        # Save checkpoint.
        if test_acc > best_acc:
            """
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            """
            best_acc = test_acc

    logger.append([000, 000, 000, 000, best_acc])
        

    logger.close()


if __name__ == '__main__':
    main()
