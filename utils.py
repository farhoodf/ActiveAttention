from __future__ import print_function, absolute_import
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res



def savefig(fname, dpi=None):
	dpi = 150 if dpi == None else dpi
	plt.savefig(fname, dpi=dpi)
	
def plot_overlap(logger, names=None):
	names = logger.names if names == None else names
	numbers = logger.numbers
	for _, name in enumerate(names):
		x = np.arange(len(numbers[name]))
		plt.plot(x, np.asarray(numbers[name]))
	return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
	'''Save training process to log file with simple plot function.'''
	def __init__(self, fpath, title=None, resume=False): 
		self.file = None
		self.resume = resume
		self.title = '' if title == None else title
		if fpath is not None:
			if resume: 
				self.file = open(fpath, 'r') 
				name = self.file.readline()
				self.names = name.rstrip().split('\t')
				self.numbers = {}
				for _, name in enumerate(self.names):
					self.numbers[name] = []

				for numbers in self.file:
					numbers = numbers.rstrip().split('\t')
					for i in range(0, len(numbers)):
						self.numbers[self.names[i]].append(numbers[i])
				self.file.close()
				self.file = open(fpath, 'a')  
			else:
				self.file = open(fpath, 'w')

	def set_names(self, names):
		if self.resume: 
			pass
		# initialize numbers as empty list
		self.numbers = {}
		self.names = names
		for _, name in enumerate(self.names):
			self.file.write(name)
			self.file.write('\t')
			self.numbers[name] = []
		self.file.write('\n')
		self.file.flush()


	def append(self, numbers):
		assert len(self.names) == len(numbers), 'Numbers do not match names'
		for index, num in enumerate(numbers):
			self.file.write("{0:.6f}".format(num))
			self.file.write('\t')
			self.numbers[self.names[index]].append(num)
		self.file.write('\n')
		self.file.flush()

	def plot(self, names=None):   
		names = self.names if names == None else names
		numbers = self.numbers
		for _, name in enumerate(names):
			x = np.arange(len(numbers[name]))
			plt.plot(x, np.asarray(numbers[name]))
		plt.legend([self.title + '(' + name + ')' for name in names])
		plt.grid(True)

	def close(self):
		if self.file is not None:
			self.file.close()


class AverageMeter(object):
	"""Computes and stores the average and current value
	   Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
	"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		
	def avg(self):
		return self.sum / self.count

def mkdir_p(path):
	'''make dir if not exist'''
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise