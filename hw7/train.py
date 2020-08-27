import os
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD , Adam
from time import time
from utils import my_argparse , get_dataloader , print_progress
from data import load_data
from model import StudentNet , encode8

# https://arxiv.org/pdf/1503.02531.pdf
def kd_loss(teacher_output , student_output , label , T = 10 , alpha = 0.5):
	hard_criterion = nn.CrossEntropyLoss()
	soft_criterion = nn.KLDivLoss(reduction = 'batchmean')
	hard_loss = hard_criterion(student_output , label)
	# https://github.com/peterliht/knowledge-distillation-pytorch/issues/2
	soft_loss = soft_criterion(F.log_softmax(student_output / T , dim = 1) , F.softmax(teacher_output / T , dim = 1))
	return (1 - alpha) * hard_loss + (alpha * T**2) * soft_loss

def knowledge_distillation(train_x , train_y , validation_x , validation_y , teacher_net , student_net , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	epoch = args.epoch

	train_loader = get_dataloader(train_x , train_y , 'train' , batch_size)
	teacher_net.to(device)
	student_net.to(device)
	teacher_net.eval()
	optimizer = Adam(student_net.parameters() , lr = learning_rate)
	max_accuracy = 0
	for i in range(epoch):
		student_net.train()
		count = 0
		total_loss = 0
		start = time()
		for (j , (data , label)) in enumerate(train_loader):
			(data , label) = (data.to(device) , label.to(device))
			optimizer.zero_grad()
			with torch.no_grad():
				teacher_output = teacher_net(data)
			student_output = student_net(data)
			(_ , index) = torch.max(student_output , dim = 1)
			count += torch.sum(label == index).item()
			loss = kd_loss(teacher_output , student_output , label)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
			end = time()
			print_progress(i + 1 , epoch , len(train_x) , batch_size , j + 1 , len(train_loader) , int(end - start) , total_loss / len(train_x) , count / len(train_x))

		if ((i + 1) % 10 == 0):
			accuracy = evaluate(validation_x , validation_y , teacher_net , student_net , device)
			if (accuracy > max_accuracy):
				max_accuracy = accuracy
				encode8(student_net.state_dict() , 'StudentNet_encode8.pkl')

	return student_net

def evaluate(validation_x , validation_y , teacher_net , student_net , device):
	validation_loader = get_dataloader(validation_x , validation_y , 'validation')
	teacher_net.to(device)
	student_net.to(device)
	teacher_net.eval()
	student_net.eval()
	count = 0
	total_loss = 0
	start = time()
	with torch.no_grad():
		for (data , label) in validation_loader:
			(data , label) = (data.to(device) , label.to(device))
			teacher_output = teacher_net(data)
			student_output = student_net(data)
			(_ , index) = torch.max(student_output , dim = 1)
			count += torch.sum(label == index).item()
			loss = kd_loss(teacher_output , student_output , label)
			total_loss += loss.item()
		end = time()
		print('evaluation ({}s) validation loss : {:.8f} , validation accuracy : {:.5f}'.format(int(end - start) , total_loss / len(validation_x) , count / len(validation_x)))
	return count / len(validation_x)

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(train_x , train_y , validation_x , validation_y) = load_data(args.train_directory , args.validation_directory , None)
	teacher_net = models.resnet18(pretrained = False , num_classes = 11)
	teacher_net.load_state_dict(torch.load('teacher_resnet18.bin' , map_location = device))
	student_net = StudentNet()
	student_net = knowledge_distillation(train_x , train_y , validation_x , validation_y , teacher_net , student_net , device , args)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)