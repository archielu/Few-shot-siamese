import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
from PIL import Image





class TrainDataset(Dataset):

	def __init__(self, dataPath, transform=None):
		super(TrainDataset, self).__init__()
		self.transform = transform
		print('Loading data into memory!')
		data_dict = {}
		idx = 0
		for degree in [0, 90, 180, 270]:
			for language_path in os.listdir(dataPath):
				for character_path in os.listdir(os.path.join(dataPath,language_path)):
					data_dict[idx] = []
					for sample_path in os.listdir(os.path.join(dataPath,language_path,character_path)):
						data_dict[idx].append(Image.open(os.path.join(dataPath,language_path,character_path,sample_path)).rotate(degree).convert('L'))

					idx += 1

		print('Finish!')
		self.datas = data_dict
		self.idx = idx

	def num_classes(self):
		return self.idx

	def __len__(self):
		return 21000000

	def __getitem__(self,index):
		## generate negative pair
		if index %2 == 0:
			label = 0
			idx1 = random.randint(0,self.idx-1)
			idx2 = random.randint(0,self.idx-1)
			while idx1 == idx2:
				idx2 = random.randint(0,self.idx-1)
			img1 = random.choice(self.datas[idx1])
			img2 = random.choice(self.datas[idx2])

		else:
			label = 1
			idx1 = random.randint(0,self.idx-1)
			img1 = random.choice(self.datas[idx1])
			img2 = random.choice(self.datas[idx1])

		if self.transform:
			img1 = self.transform(img1)
			img2 = self.transform(img2)

		return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))



class TestDataset(Dataset):
	def __init__(self, dataPath, group_num, group_size, transform=None):
		super(TestDataset, self).__init__()
		self.group_num = group_num
		self.group_size = group_size
		print('Loading testing dataset to memory')
		data_dict = {}
		idx = 0
		for language_path in os.listdir(dataPath):
			for character_path in os.listdir(os.path.join(dataPath,language_path)):
				data_dict[idx] = []
				for sample_path in os.listdir(os.path.join(dataPath,language_path,character_path)):
					data_dict[idx].append(Image.open(os.path.join(dataPath,language_path,character_path,sample_path)).convert('L'))
				idx += 1

		print('Finish!')
		self.datas = data_dict
		self.num_classes = idx
		self.transform = transform

	def __len__(self):
		return self.group_num * self.group_size

	def __getitem__(self, index):
		## Generate positive pair
		if index % self.group_size == 0:
			idx = random.randint(0,self.num_classes-1)
			img1 = random.choice(self.datas[idx])
			img2 = random.choice(self.datas[idx])
		else:
			idx1 = random.randint(0,self.num_classes-1)
			idx2 = random.randint(0,self.num_classes-1)
			while idx1 == idx2:
				idx2 = random.randint(0,self.num_classes-1)

			img1 = random.choice(self.datas[idx1])
			img2 = random.choice(self.datas[idx2])

		if self.transform:
			img1 = self.transform(img1)
			img2 = self.transform(img2)


		return img1, img2








