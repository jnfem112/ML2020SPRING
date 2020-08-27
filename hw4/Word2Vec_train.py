from gensim.models import Word2Vec
from utils import my_argparse
from data import load_vocabulary

def main(args):
	vocabulary = load_vocabulary(args.train_data , args.nolabel_data , args.test_data)
	W2Vmodel = Word2Vec(vocabulary , size = args.embedding_dimension , sg = 1 , window = 5 , min_count = 5 , iter = args.iteration , workers = 12)
	W2Vmodel.save('Word2Vec.model')
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)