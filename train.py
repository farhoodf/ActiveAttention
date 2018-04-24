from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import shutil
import datetime
import random

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

from torch.autograd import Variable

from vgg import VGG
from dataset import getdata

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--step', type=int, default=25, help='Decrease learning rate after number of epochs.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--attention', default='no-att', type=str, help='attention, options:no-att,')
parser.add_argument('--nclass', default=10, type=int, help='Number of classes')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer, options: sgd, adam,')

parser.add_argument('--cpu', action='store_false', dest='gpu', help='Use Cuda')


# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, help='evaluate model on validation set')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}


def train(model, trainloader, criterion, optimizer, epoch, use_cuda):
	model.train()

	losses = AverageMeter()
	top1 = AverageMeter()

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda(async=True)
		inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

		outputs = model(inputs)
		loss = criterion(outputs, targets)

		prec1 = accuracy(outputs.data, targets.data, topk=(1,))
		losses.update(loss.data[0], inputs.size(0))
		top1.update(prec1[0], inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


	return (losses.avg(), top1.avg())

def test(model, testloader, criterion, use_cuda):

	model.eval()

	losses = AverageMeter()
	top1 = AverageMeter()


	for batch_idx, (inputs, targets) in enumerate(testloader):
		

		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

		outputs = model(inputs)
		loss = criterion(outputs, targets)

		prec1 = accuracy(outputs.data, targets.data, topk=(1,))
		losses.update(loss.data[0], inputs.size(0))
		top1.update(prec1[0], inputs.size(0))


	return (losses.avg(), top1.avg())

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
	filename = state['attention'] + '-' + filename
	filepath = os.path.join(checkpoint, filename)
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(checkpoint, state['attention'] + '-' +'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
	global state
	if (epoch+1) % args.step == 0:
		state['lr'] *= args.gamma
		for param_group in optimizer.param_groups:
			param_group['lr'] = state['lr']

def main():
	best_acc = 0
	start_epoch = args.start_epoch

	if not os.path.isdir(args.checkpoint):
		mkdir_p(args.checkpoint)

	trainloader = getdata(args, train=True)
	testloader = getdata(args, train=False)

	model = VGG(args.attention, args.nclass)

	if args.gpu:
		if torch.cuda.is_available():
			model = model.cuda()
			cudnn.benchmark = True
		else:
			print ('There is no cuda available on this machine use cpu instead.')
			args.gpu = False

	criterion = nn.CrossEntropyLoss()
	optimizer = ''
	if args.optimizer == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	else:
		print (args.optimizer,'is not correct')
		return

	title = 'cifar-10-' + args.attention
	
	if args.evaluate:
		print('\nEvaluation only')
		assert os.path.isfile(args.evaluate), 'Error: no checkpoint directory found!'
		checkpoint = torch.load(args.evaluate)
		model.load_state_dict(checkpoint['state_dict'])
		test_loss, test_acc = test(model, testloader, criterion, args.gpu)
		print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
		return

	if args.resume:
		# Load checkpoint.
		print('==> Resuming from checkpoint..')
		assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
		args.checkpoint = os.path.dirname(args.resume)
		checkpoint = torch.load(args.resume)
		best_acc = checkpoint['best_acc']
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		logger = Logger(os.path.join(args.checkpoint, state['attention'] + '-' +'log.txt'), title=title, resume=True)
	else:
		logger = Logger(os.path.join(args.checkpoint, state['attention'] + '-' +'log.txt'), title=title)
		logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


	for epoch in range(start_epoch, args.epochs):
		start_time = time.time()
		adjust_learning_rate(optimizer, epoch)

		train_loss, train_acc = train(model, trainloader, criterion, optimizer, epoch, args.gpu)
		test_loss, test_acc = test(model, testloader, criterion, args.gpu)
		if sys.version[0] == '3':
			train_acc = train_acc.cpu().numpy().tolist()[0]
			test_acc = test_acc.cpu().numpy().tolist()[0]
		logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

		is_best = test_acc > best_acc
		best_acc = max(test_acc, best_acc)
		save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'acc': test_acc,
				'best_acc': best_acc,
				'optimizer' : optimizer.state_dict(),
				'attention' : state['attention'],
				}, is_best, checkpoint=args.checkpoint)
		print (	time.time() - start_time)
		print("epoch: {:3d}, lr: {:.8f}, train-loss: {:.3f}, test-loss: {:.3f}, train-acc: {:2.3f}, test_acc:, {:2.3f}".format(
				epoch,state['lr'],train_loss,test_loss,train_acc,test_acc))

	logger.close()
	logger.plot()
	savefig(os.path.join(args.checkpoint, state['attention'] + '-' + 'log.eps'))

	print('Best acc:',best_acc)

if __name__ == '__main__':
	main()
