from src.dataset import TrainDataset, TestDataset
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
from torchvision import transforms
import torchvision.datasets as dset
import numpy as np
from argparse import ArgumentParser, Namespace
from src.loss import BCELoss 
from collections import deque
from pathlib import Path
import time
import os
from src.model import Siamese
from collections import deque
import pickle
import torchvision
import matplotlib.pyplot as plt
import torch



def parse_args() -> Namespace:
	parser = ArgumentParser()
	parser.add_argument("--train_path",help="Directory to the train dataset.",default="./Omniglot/images_background")
	parser.add_argument("--test_path",help="Directory to the test dataset.",default="./Omniglot/images_evaluation")
	parser.add_argument("--group_num",type=int,help="number of groups.",default=400)
	parser.add_argument("--group_size", type=int, help="number of data in one group",default=20)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--lr", type=int, default=0.0006)
	parser.add_argument("--show_every", type=int, default=10)
	parser.add_argument("--save_every", type=int, default=100)
	parser.add_argument("--test_every", type=int, default=100)
	parser.add_argument("--max_iter",type=int, default = 50000)
	parser.add_argument("--model_path",type=Path, default = './model_checkpoint')


	args = parser.parse_args()
	return args



if __name__ == '__main__':
	args = parse_args()
	os.makedirs(args.model_path, exist_ok=True)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_transforms = transforms.Compose([
		transforms.RandomAffine(15),
		transforms.ToTensor()
	])

	train_dataset = TrainDataset(args.train_path, transform = data_transforms)
	test_dataset = TestDataset(args.test_path, group_num = args.group_num, group_size = args.group_size, transform = transforms.ToTensor())

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size = args.group_size, shuffle=False)

	criterion = BCELoss()
	model = Siamese()
	model.to(device)

	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

	train_losses = []
	accuracy_log = []
	accuracies = []
	batch_loss = 0
	queue = deque(maxlen=20)
	time_start = time.time()

	for batch_idx, (img1,img2,labels) in enumerate(train_loader):
		if batch_idx >= args.max_iter:
			break
		img1 = img1.to(device)
		img2 = img2.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		output = model.forward(img1,img2)
		loss = criterion(output, labels)
		batch_loss += loss.item()
		current_loss = loss.item()
		loss.backward()
		optimizer.step()

		if batch_idx % args.show_every == 0:
			print('[Iter : {0}]     loss : {1}     Time : {2}'.format(batch_idx, batch_loss, time.time() - time_start))
			batch_loss = 0
			time_start = time.time()

		if batch_idx % args.save_every == 0:
			torch.save(model.state_dict(), args.model_path/'checkpoint.pt')

		if batch_idx % args.test_every == 0:
			right = 0
			error = 0
			for _, (test1, test2) in enumerate(test_loader):
				test1 = test1.to(device)
				test2 = test2.to(device)
				result = model(test1,test2).detach().cpu().numpy()
				pred = np.argmax(result)
				if pred == 0:
					right += 1
				else:
					error += 1

			print('*' * 70)
			print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_idx, right, error, right*1.0/(right+error)))
			print('*' * 70)
			queue.append(right*1.0/(right+error))
			accuracy_log.append(right*1.0/(right+error))
			train_losses.append(current_loss)


	with open('train_loss', 'wb') as f:
		pickle.dump(train_losses, f)

	with open('test_accuracy', 'wb') as f:
		pickle.dump(accuracy_log, f)

	acc = 0.0
	for d in queue:
		acc += d
	print("#"*70)
	print("final accuracy: ", acc/20)






		





	

	
		





	

	