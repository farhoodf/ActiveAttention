import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def getdata(args, train=True):
	if args.dataset == 'cifar10':
		datasetloader = datasets.CIFAR10
	elif args.dataset == 'cifar100':
		datasetloader = datasets.CIFAR100

	if train:
		transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		ds =datasetloader(root='./data', train=train, download=True, transform=transform)
		dataloader = data.DataLoader(ds, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

		return dataloader
	else:
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		ds = datasetloader(root='./data', train=train, download=False, transform=transform)
		dataloader = data.DataLoader(ds, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

		return dataloader
