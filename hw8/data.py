import os
import numpy as np
from utils import Dictionary

def preprocess(data , token2index , max_length):
	data = data.split()
	data = [token2index.get(token , token2index['<UNK>']) for token in data]
	data.insert(0 , token2index['<BOS>'])
	data.append(token2index['<EOS>'])
	data += (max_length - len(data)) * [token2index['<PAD>']]
	return data

def deprocess(data , index2token):
	sentence = list()
	for index in data:
		token = index2token[str(index.item())]
		if (token == '<EOS>'):
			break
		else:
			sentence.append(token)
	return sentence

def load_data(path , dictionary , max_length):
	x = list()
	y = list()
	with open(path , 'r') as file:
		for line in file:
			(english , chinese) = line.rstrip('\n').split('\t')
			english = preprocess(english , dictionary.token2index_english , max_length)
			chinese = preprocess(chinese , dictionary.token2index_chinese , max_length)
			x.append(english)
			y.append(chinese)
	return (np.array(x) , np.array(y))

def load_dataset(root , mode , max_length):
	dictionary = Dictionary(root)

	if (mode == 'train'):
		(train_x , train_y) = load_data(os.path.join(root , 'training.txt') , dictionary , max_length)
		(validation_x , validation_y) = load_data(os.path.join(root , 'validation.txt') , dictionary , max_length)
		return (train_x , train_y , validation_x , validation_y , dictionary)
	else:
		(test_x , test_y) = load_data(os.path.join(root , 'testing.txt') , dictionary , max_length)
		return (test_x , test_y , dictionary)

def dump(predict , output_file):
	with open(output_file , 'w') as file:
		file.writelines(predict)
	return