import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	args = parser.parse_args()
	return args

class Dataset(Dataset):
	def __init__(self , directory , way , query):
		self.character_list = [character_path for character_path in glob(os.path.join(directory , '**/character*') , recursive = True)]
		self.transform = transforms.ToTensor()
		self.way = way
		self.query = query
		return

	def __len__(self):
		return len(self.character_list)

	def __getitem__(self , index):
		sample = np.arange(20)
		np.random.shuffle(sample)
		character_path = self.character_list[index]
		image_list = [character for character in glob(os.path.join(character_path , '**/*.png') , recursive = True)]
		image_list.sort()
		image = [self.transform(Image.open(image_name)) for image_name in image_list]
		image = torch.stack(image , dim = 0)
		data = image[sample[ : self.way + self.query]]
		return data	

def get_meta_batch(way , shot , query , meta_batch_size , dataloader , iterator):
	data = list()
	for i in range(meta_batch_size):
		try:
			temp = iterator.next()
		except StopIteration:
			iterator = iter(data_loader)
			temp = iterator.next()
		train_data = temp[ : , : shot].reshape(-1 , 1 , 28 , 28)
		validation_data = temp[ : , shot : ].reshape(-1 , 1 , 28 , 28)
		data.append(torch.cat((train_data , val_data) , dim = 0))
	data = torch.stack(data , dim = 0)
	return (data , iterator)

def create_label(way , shot):
	return torch.arange(way).repeat_interleave(shot)

def ConvBlock(in_channels , out_channels):
	return nn.Sequential(
				nn.Conv2d(in_channels , out_channels , kernel_size = 3 , padding = 1) ,
				nn.BatchNorm2d(out_channels) ,
				nn.ReLU() ,
				nn.MaxPool2d(kernel_size = 2 , stride = 2)
			)

def ConvBlockFunction(x , w , b , w_bn , b_bn):
	x = F.conv2d(x , w , b , padding = 1)
	x = F.batch_norm(x , running_mean = None , running_var = None , weight = w_bn , bias = b_bn , training = True)
	x = F.relu(x)
	x = F.max_pool2d(x , kernel_size = 2 , stride = 2)
	return x

class Classifier(nn.Module):
	def __init__(self , output_dim):
		super(Classifier , self).__init__()
		self.conv1 = ConvBlock(1 , 64)
		self.conv2 = ConvBlock(64 , 64)
		self.conv3 = ConvBlock(64 , 64)
		self.conv4 = ConvBlock(64 , 64)
		self.fc1 = nn.Linear(64 , output_dim)
		return
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.view(x.size(dim = 0) , -1)
		x = self.fc1(x)
		return x

	def functional_forward(self , x , params):
		for i in range(4):
			x = ConvBlockFunction(x , params[f'conv{i + 1}.0.weight'] , params[f'conv{i + 1}.0.bias'] , params.get(f'conv{i + 1}.1.weight') , params.get(f'conv{i + 1}.1.bias'))
		x = x.view(x.size(dim = 0) , -1)
		x = F.linear(x , params['fc1.weight'] , params['fc1.bias'])
		return x

def MAML(data , model , optimizer , criterion , device , args , train = True):
	# Hyper-parameter.

	model.to(device)
	model.train()
	loss_list = list()
	accuracy_list = list()
	for x in data:
		train_x = x[ : way * shot]
		validation_x = x[way * shot : ]
		train_y = create_label(way , shot)
		validation_y = create_label(way , shot)
		(train_x , train_y) = (train_x.to(device , dtype = torch.float) , train_y.to(device , dtype = torch.long))
		(validation_x , validation_y) = (validation_x.to(device , dtype = torch.float) , validation_y.to(device , dtype = torch.long))
		fast_weights = OrderedDict(model.named_parameters())
		for i in range(inner_step):
			output = model.functional_forward(train_x , fast_weights)
			loss = criterion(output , train_y)
			grads = torch.autograd.grad(loss , fast_weights.values() , create_graph = True)
			fast_weights = OrderedDict((name , param - inner_lr * grad) for ((name , param) , grad) in zip(fast_weights.items() , grads))
		output = model.functional_forward(validation_x , fast_weights)
		loss = criterion(output , validation_y)
		loss_list.append(loss)
		index = torch.argmax(output , dim = -1)
		accuracy = torch.mean(index == validation_y)
		accuracy_list.append(accuracy)
	loss = torch.mean(loss , dim = 0)
	accuracy = torch.mean(accuracy , dim = 0)

	optimizer.zero_grad()
	if (train):
		loss.backward()
		optimizer.step()

	return (loss.item() , accuracy.item())

def train():
	# Hyper-parameter.
	way = args.way
	shot = args.shot
	query = args.query
	learning_rate = args.learning_rate
	epoch = args.epoch

	train_dataset = Dataset(args.train_directory , shot , query)
	(train_dataset , validation_dataset) = random_split(train_dataset , [3200 , 656])
	train_dataloader = DataLoader(train_dataset , batch_size = way , drop_last = True , shuffle = True , num_workers = 8)
	validation_dataloader = DataLoader(validation_dataset , batch_size = way , drop_last = True , shuffle = True , num_workers = 8)
	train_iterator = iter(train_dataloader)
	validation_iterator = iter(val_dataloader)
	optimizer = Adam(model.parameters() , lr = learning_rate)
	criterion = nn.CrossEntropyLoss()
	for i in range(epoch):
		loss_list = list()
		accuracy_list = list()
		for j in range(len(train_dataloader) // meta_batch_size):
			(data , train_iterator) = get_meta_batch(way , shot , query , meta_batch_size , train_dataloader , train_iterator)
			(loss , accuracy) = MAML(data , model , optimizer , criterion , device , args)
			loss_list.append(loss)
			accuracy_list.append(accuracy)

		loss_list = list()
		accuracy_list = list()
		for j in range(len(validation_dataloader) // batch_size):
			(data , validation_iterator) = get_meta_batchget_meta_batch(way , shot , query , batch_size , validation_dataloader , validation_iterator)
			(loss , accuracy) = MAML(data , model , optimizer , criterion , device , args , train = False)
			loss_list.append(loss)
			accuracy_list.append(accuracy)

	return model

def test():
	# Hyper-parameter.

	test_dataset = Dataset(args.test_directory , shot , query)
	test_dataloader = DataLoader(test_dataset , batch_size = way , drop_last = True , shuffle = True , num_workers = 8)
	test_iterator = iter(test_dataloader)

	loss_list = list()
	accuracy_list = list()
	for i in range(len(test_dataloader) // batch_size):
		(data , test_iterator) = get_meta_batch(way , shot , query , batch_size , test_dataloader , test_iterator)
		(loss , accuracy) = MAML(data , model , optimizer , criterion , device , args , train = False)
			loss_list.append(loss)
			accuracy_list.append(accuracy)

	return

def main(args):
	print('============ setup device ============')
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('============ create model ============')
	meta_model = Classifier(args.way)
	print('=============== train ================')
	train()
	print('================ test ================')
	test()
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)