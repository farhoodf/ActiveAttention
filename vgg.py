import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

class VGG(nn.Module):

	def __init__(self, attn='0', num_classes=10):
		super(VGG, self).__init__()

		self.attn_type = attn

		self.cfgs = {
				'part1': [64,'D',64,128,'D',128,256,'D',256,'D',256],
				'part2': ['M',512,'D',512,'D',512],
				'part3': ['M',512,'D',512,'D',512],
				'part4': ['M',512,'D','M',512,'M']
			}

		self._make_layers(batch_norm=True)
		self._make_classifier(num_classes)
		self._make_compat_pc()
		self._initialize_weights()

		self.softmaxattn = nn.Softmax(dim=1)

	def forward(self, x):
		loc7 = self.part1(x)
		loc10 = self.part2(loc7)
		loc13 = self.part3(loc10)

		glob = self.part4(loc13).view(loc13.size(0), -1)
		glob = self.part5(glob)

		# #without att
		# out = self.classifier(glob)
		# return out
		if self.attn_type == 'no-att':
			out = self.classifier(glob)
			return out

		if self.attn_type == '1dp':
			out = self.forward_attdp(glob, loc13)
			out = self.classifier(out)
			return out

		if self.attn_type == '1pc':
			out = self.forward_attpc(glob, loc13, self.pcConv13)
			out = self.classifier(out)
			return out

		if self.attn_type == '2dp':
			at1 = self.forward_attdp(glob, loc13)
			at2 = self.forward_attdp(glob, loc10)
			out = torch.cat([at1,at2],dim=1)
			out = self.classifier(out)
			return out

		if self.attn_type == '2pc':
			at1 = self.forward_attpc(glob, loc13, self.pcConv13)
			at2 = self.forward_attpc(glob, loc10, self.pcConv10)
			out = torch.cat([at1,at2],dim=1)
			out = self.classifier(out)
			return out

		if self.attn_type == '2dp-ind':
			at13 = self.forward_attdp(glob, loc13)
			at10 = self.forward_attdp(glob, loc10)

			out1 = self.classifier13(at13)
			out2 = self.classifier13(at10)
			out = out1.add(out2)
			return out

		if self.attn_type == '2pc-ind':
			at13 = self.forward_attpc(glob, loc13, self.pcConv13)
			at10 = self.forward_attpc(glob, loc10, self.pcConv10)
			
			out1 = self.classifier13(at13)
			out2 = self.classifier13(at10)
			out = out1.add(out2)
			return out


		return

	def forward_attdp(self, glob, loc):
		f = loc.size(-1)

		glob_ = glob.repeat(f,1,1).repeat(f,1,1,1).transpose(0,2).transpose(1,3)

		c = glob_.mul(loc).sum(dim=1).view(-1,f*f)
		c = self.softmaxattn(c)
		c = c.repeat(512,1,1).transpose(0,1)

		loc_attn = c.mul(loc.view(-1,512,f*f)).sum(dim=2)

		return loc_attn

	def forward_attpc(self, glob, loc, cnv):
		f = loc.size(-1)

		glob_ = glob.repeat(f,1,1).repeat(f,1,1,1).transpose(0,2).transpose(1,3)

		c = cnv(glob_.add(loc)).view(-1,f*f)
		c = self.softmaxattn(c)
		c = c.repeat(512,1,1).transpose(0,1)

		loc_attn = c.mul(loc.view(-1,512,f*f)).sum(dim=2)

		return loc_attn


	# def forward_att1dp(self, glob, loc):

	# 	glob = glob.repeat(8,1,1).repeat(8,1,1,1).transpose(0,2).transpose(1,3)

	# 	c = glob.mul(loc).sum(dim=1).view(-1,64)
	# 	c = self.softmax(c)
	# 	c = c.repeat(512,1,1).transpose(0,1)

	# 	loc_attn = c.mul(loc.view(-1,512,64)).sum(dim=2)

	# 	return loc_attn

	# def forward_att2dp(self, glob, loc):

	# 	glob = glob.repeat(16,1,1).repeat(16,1,1,1).transpose(0,2).transpose(1,3)

	# 	c = glob.mul(loc).sum(dim=1).view(-1,256)
	# 	c = self.softmax(c)
	# 	c = c.repeat(512,1,1).transpose(0,1)

	# 	loc_attn = c.mul(loc.view(-1,512,256)).sum(dim=2)

	# 	return loc_attn
	
	# def forward_att1pc(self, glob, loc):
	# 	glob = glob.repeat(8,1,1).repeat(8,1,1,1).transpose(0,2).transpose(1,3)

	# 	c = glob.add(loc).sum(dim=1).view(-1,64)
	# 	c = self.softmax(c)
	# 	c = c.repeat(512,1,1).transpose(0,1)

	# 	loc_attn = c.mul(loc.view(-1,512,64)).sum(dim=2)

	# 	return loc_attn

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def _make_classifier(self, num_classes):
		num_feat = 512
		if 'ind' not in self.attn_type:
			if '2' in self.attn_type:
				num_feat = 1024
			layers = []
			layers += [nn.BatchNorm1d(num_feat)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(num_feat, num_classes)]
			self.classifier = nn.Sequential(*layers)
		
		elif 'ind' in self.attn_type:
			layers = []
			layers += [nn.BatchNorm1d(num_feat)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(num_feat, num_classes)]
			layers += [nn.Softmax(dim=1)]
			self.classifier13 = nn.Sequential(*layers)

			layers = []
			layers += [nn.BatchNorm1d(num_feat)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(num_feat, num_classes)]
			layers += [nn.Softmax(dim=1)]
			self.classifier10 = nn.Sequential(*layers)

	def _make_compat_pc(self):
		if 'pc' in self.attn_type:
			self.pcConv13 = nn.Conv2d(512, 1, kernel_size=1, padding=0)
			self.pcConv10 = nn.Conv2d(512, 1, kernel_size=1, padding=0)
			self.pcConv7 = nn.Conv2d(512, 1, kernel_size=1, padding=0)

	def _make_layers(self, batch_norm = True):
		firstDP = True
		layers = []
		in_channels = 3
		for v in self.cfgs['part1']:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			elif v == 'D':
				if firstDP:
					layers += [nn.Dropout(0.3)]
					firstDP = False
				else:
					layers += [nn.Dropout(0.4)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v

		self.part1 = nn.Sequential(*layers)
		layers = []
		for v in self.cfgs['part2']:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			elif v == 'D':
				layers += [nn.Dropout(0.4)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v

		self.part2 = nn.Sequential(*layers)

		layers = []
		for v in self.cfgs['part3']:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			elif v == 'D':
				layers += [nn.Dropout(0.4)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v

		self.part3 = nn.Sequential(*layers)

		layers = []
		for v in self.cfgs['part4']:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			elif v == 'D':
				layers += [nn.Dropout(0.4)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v

		self.part4 = nn.Sequential(*layers)

		layers = [nn.Linear(512, 512)]
		self.part5 = nn.Sequential(*layers)

