import torch
import torch.nn as nn
import torch.nn.functional as F 


class Siamese(nn.Module):
	def __init__(self):
		super(Siamese,self).__init__()
		self.Conv = nn.Sequential(
			nn.Conv2d(1,64,10),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(2),
			nn.Conv2d(64,128,7),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(128,128,4),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(128,256,4),
			nn.ReLU(inplace=True),
			)

		self.fcn_1= nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
		self.fcn_2 = nn.Sequential(nn.Linear(4096,1), nn.Sigmoid())


	def siamese_distance(self,x1,x2):
		x1 = self.Conv(x1)
		x1 = x1.view(x1.size()[0], -1)
		x1 = self.fcn_1(x1)
		x2 = self.Conv(x2)
		x2 = x2.view(x2.size()[0], -1)
		x2 = self.fcn_1(x2)

		dis = torch.abs(x1-x2)

		return dis
	
	def forward(self,x1,x2):
		dis = self.siamese_distance(x1,x2)
		out = self.fcn_2(dis)

		return out



