import os
import torch
import torch.nn as nn
from time import time
from utils import my_argparse , get_dataloader , BLEU
from data import load_dataset , deprocess , dump
from model import Encoder , Decoder , Seq2Seq

def test(test_x , test_y , dictionary , model , device , args):
	test_loader = get_dataloader(test_x , test_y , 'test' , batch_size = 1)
	model.to(device)
	model.eval()
	criterion = nn.CrossEntropyLoss(ignore_index = dictionary.token2index_chinese['<PAD>'])
	total_loss = 0
	total_score = 0
	predicts = list()
	start = time()
	with torch.no_grad():
		for (input , target) in test_loader:
			(input , target) = (input.to(device) , target.to(device))
			(output , predict) = model.inference(input , target , args.beam_size)
			output = output.reshape(-1 , output.size(dim = 2))
			target = target[ : , 1 : ].reshape(-1)
			loss = criterion(output , target)
			total_loss += loss.item()
			(predict , target) = (predict.squeeze(dim = 0) , target.squeeze(dim = 0))
			predict = deprocess(predict , dictionary.index2token_chinese)
			target = deprocess(target , dictionary.index2token_chinese)
			total_score += BLEU(predict , target)
			predicts.append(''.join(predict) + '\n')
	end = time()
	print('evaluation ({}s) test loss : {:.8f} , test BLEU : {:.5f}'.format(int(end - start) , total_loss / test_x.shape[0] , total_score / test_x.shape[0]))
	return predicts

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	(test_x , test_y , dictionary) = load_dataset(args.root , 'test' , args.max_length)
	encoder = Encoder(len(dictionary.token2index_english) , embedding_dim = 256 , hidden_dim = 512 , num_layers = 3 , dropout = 0.5)
	decoder = Decoder(len(dictionary.token2index_chinese) , embedding_dim = 256 , hidden_dim = 1024 , num_layers = 3 , dropout = 0.5)
	model = Seq2Seq(encoder , decoder)
	model.load_state_dict(torch.load('Seq2Seq.pkl' , map_location = device))
	predict = test(test_x , test_y , dictionary , model , device , args)
	dump(predict , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)