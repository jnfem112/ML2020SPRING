import os
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from utils import my_argparse , get_dataloader
from data import load_data
from model import Autoencoder

def cluster(dataset , model , device , args):
	dataloader = get_dataloader(dataset , 'cluster')
	model.to(device)
	model.eval()
	new_dataset = list()
	with torch.no_grad():
		for data in dataloader:
			data = data.to(device)
			(encode , decode) = model(data)
			new_dataset.append(encode.cpu().detach().numpy())
	new_dataset = np.concatenate(new_dataset , axis = 0)
	new_dataset = new_dataset.reshape(dataset.shape[0] , -1)
	new_dataset = (new_dataset - np.mean(new_dataset , axis = 0)) / (np.std(new_dataset , axis = 0) + 1e-8)

	minibatch_kmeans = MiniBatchKMeans(n_clusters = args.n_clusters , batch_size = 100)
	minibatch_kmeans = minibatch_kmeans.fit(new_dataset)
	label = minibatch_kmeans.predict(new_dataset)
	score = np.sum(np.square(minibatch_kmeans.cluster_centers_[label] - new_dataset) , axis = 1)

	return score

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	test_x = load_data(args.test_data)
	model = Autoencoder()
	model.load_state_dict(torch.load('Autoencoder.pkl' , map_location = device))
	test_y = cluster(test_x , model , device , args)
	dump(test_y , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)