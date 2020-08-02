import argparse
from gensim.models import Word2Vec

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data' , type = str)
	parser.add_argument('--nolabel_data' , type = str)
	parser.add_argument('--test_data' , type = str)
	parser.add_argument('--embedding_dimension' , type = int)
	parser.add_argument('--iteration' , type = int)
	args = parser.parse_args()
	return args

def load_data(args):
	def valid(token):
		return all(character.isalnum() for character in token)

	vocabulary = list()

	with open(args.train_data , 'r') as file:
		for line in file:
			line = line.rstrip('\n').split()[2 : ]
			line = [token for token in line if valid(token)]
			vocabulary.append(line)

	with open(args.nolabel_data , 'r') as file:
		for line in file:
			line = line.rstrip('\n').split()
			line = [token for token in line if valid(token)]
			vocabulary.append(line)

	with open(args.validation_data , 'r') as file:
		next(file)
		for line in file:
			line = ','.join(line.split(',')[1 : ])
			line = line.rstrip('\n').split()
			line = [token for token in line if valid(token)]
			vocabulary.append(line)

	return vocabulary

def main(args):
	vocabulary = load_data(args)
	W2Vmodel = Word2Vec(vocabulary , size = args.embedding_dimension , sg = 1 , window = 5 , min_count = 5 , iter = args.iteration , workers = 12)
	W2Vmodel.save('Word2Vec.model')
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)
