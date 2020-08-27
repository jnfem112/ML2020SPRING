import argparse
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader

class Dataset(Dataset):
	def __init__(self , data , label , transform):
		self.data = data
		self.label = label
		self.transform = transform
		return

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self , index):
		if (self.label is not None):
			return (self.transform(self.data[index]) , self.label[index])
		else:
			return self.transform(self.data[index])

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda' , type = str , default = '0')
	parser.add_argument('--train_directory' , type = str , default = 'food-11/training/')
	parser.add_argument('--validation_directory' , type = str , default = 'food-11/validation/')
	parser.add_argument('--test_directory' , type = str , default = 'food-11/testing/')
	parser.add_argument('--output_file' , type = str , default = 'predict.csv')
	parser.add_argument('--batch_size' , type = int , default = 128)
	parser.add_argument('--learning_rate' , type = float , default = 0.001)
	parser.add_argument('--epoch' , type = int , default = 150)
	args = parser.parse_args()
	return args

def get_transform(mode):
	if (mode == 'train'):
		return transforms.Compose([
			transforms.ToPILImage() ,
			transforms.RandomAffine(15 , translate = (0.1 , 0.1) , scale = (0.9 , 1.1)) ,
			transforms.RandomHorizontalFlip() ,
			transforms.ToTensor()
		])
	else:
		return transforms.Compose([
			transforms.ToTensor()
		])

def get_dataloader(data , label , mode , batch_size = 1024 , num_workers = 8):
	transform = get_transform(mode)
	dataset = Dataset(data , label , transform)
	dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = (mode == 'train') , num_workers = num_workers)
	return dataloader

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , loss = None , accuracy = None):
	if (batch < total_batch):
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('epoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '\r')
	else:
		data = total_data
		bar = 50 * '='
		print('epoch {}/{} ({}/{}) [{}] ({}s) train loss : {:.8f} , train accuracy : {:.5f}'.format(epoch , total_epoch , data , total_data , bar , total_time , loss , accuracy))
	return