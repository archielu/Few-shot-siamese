import torch
import torch.nn as nn
import torch.nn.functional as F 



class BCELoss(nn.Module):
	def __init__(self):
		super(BCELoss,self).__init__()

	def forward(self, output, target):
		criterion = nn.BCELoss()
		loss = criterion(output,target)
		return loss
