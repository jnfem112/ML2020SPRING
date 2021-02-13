import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from time import time
from utils import my_argparse , get_dataloader , print_progress
from model import Generator , Classifier

def discrepancy(output_1 , output_2):
	return torch.mean(torch.abs(F.softmax(output_1 , dim = 1) - F.softmax(output_2 , dim = 1)))

def train(generator , classifier_1 , classifier_2 , device , args):
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	weight_decay = args.weight_decay
	epoch = args.epoch

	(source_dataloader , target_dataloader) = get_dataloader(args.directory , 'train' , batch_size)
	number_of_batch = min(len(source_dataloader) , len(target_dataloader))
	generator.to(device)
	classifier_1.to(device)
	classifier_2.to(device)
	optimizer_generator = Adam(generator.parameters() , lr = learning_rate , weight_decay = weight_decay)
	optimizer_classifier_1 = Adam(classifier_1.parameters() , lr = learning_rate , weight_decay = weight_decay)
	optimizer_classifier_2 = Adam(classifier_2.parameters() , lr = learning_rate , weight_decay = weight_decay)
	for i in range(epoch):
		generator.train()
		classifier_1.train()
		classifier_2.train()
		start = time()
		for (j , ((source_data , source_label) , (target_data , _))) in enumerate(zip(source_dataloader , target_dataloader)):
			(source_data , source_label , target_data) = (source_data.to(device) , source_label.to(device) , target_data.to(device))
			# Step 1
			optimizer_generator.zero_grad()
			optimizer_classifier_1.zero_grad()
			optimizer_classifier_2.zero_grad()
			feature = generator(source_data)
			output_1 = classifier_1(feature)
			output_2 = classifier_2(feature)
			loss = F.cross_entropy(output_1 , source_label) + F.cross_entropy(output_2 , source_label)
			loss.backward()
			optimizer_generator.step()
			optimizer_classifier_1.step()
			optimizer_classifier_2.step()
			# Step 2
			optimizer_generator.zero_grad()
			optimizer_classifier_1.zero_grad()
			optimizer_classifier_2.zero_grad()
			feature = generator(source_data)
			output_1 = classifier_1(feature)
			output_2 = classifier_2(feature)
			loss_1 = F.cross_entropy(output_1 , source_label) + F.cross_entropy(output_2 , source_label)
			feature = generator(target_data)
			output_1 = classifier_1(feature)
			output_2 = classifier_2(feature)
			loss_2 = discrepancy(output_1 , output_2)
			loss = loss_1 - loss_2
			loss.backward()
			optimizer_classifier_1.step()
			optimizer_classifier_2.step()
			# Step 3
			for k in range(4):
				feature = generator(target_data)
				output_1 = classifier_1(feature)
				output_2 = classifier_2(feature)
				loss = discrepancy(output_1 , output_2)
				loss.backward()
				optimizer_generator.step()
			end = time()
			print_progress(i + 1 , epoch , j + 1 , number_of_batch , int(end - start))

	return (generator , classifier_1 , classifier_2)

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(generator , classifier_1 , classifier_2) = (Generator() , Classifier() , Classifier())
	(generator , classifier_1 , classifier_2) = train(generator , classifier_1 , classifier_2 , device , args)
	torch.save(generator.state_dict() , 'generator.pkl')
	torch.save(classifier_1.state_dict() , 'classifier_1.pkl')
	torch.save(classifier_2.state_dict() , 'classifier_2.pkl')
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)