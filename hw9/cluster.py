import os
import numpy as np
import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from utils import my_argparse , set_random_seed , get_dataloader
from data import load_data
from model import Autoencoder

def cluster(dataset , model , device , args):
	set_random_seed(0)

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

	kernel_pca = KernelPCA(n_components = 200 , kernel = 'rbf' , n_jobs = -1)
	new_dataset = kernel_pca.fit_transform(new_dataset)

	tsne = TSNE(n_components = 2)
	new_dataset = tsne.fit_transform(new_dataset)

	minibatch_kmeans = MiniBatchKMeans(n_clusters = 2 , random_state = 0)
	minibatch_kmeans.fit(new_dataset)
	predict = minibatch_kmeans.labels_

	return predict

def main(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_x = load_data(args.train_x)
	model = Autoencoder()
	model.load_state_dict(torch.load(args.checkpoint , map_location = device))
	train_y = cluster(train_x , model , device , args)
	dump(train_y , args.output_file)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)