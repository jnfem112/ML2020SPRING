import argparse

def my_argparse():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data' , type = str , default = 'train.csv')
	parser.add_argument('--test_data' , type = str , default = 'test.csv')
	parser.add_argument('--output_file' , type = str , default = 'predict.csv')
	parser.add_argument('--batch_size' , type = int , default = 1024)
	parser.add_argument('--learning_rate' , type = float , default = 0.001)
	parser.add_argument('--lambd' , type = float , default = 0.001)
	parser.add_argument('--epoch' , type = int , default = 2000)
	args = parser.parse_args()
	return args

def print_progress(epoch , total_epoch , total_data , batch_size , batch , total_batch , total_time = None , train_RMSE = None , validation_RMSE = None):
	if (batch < total_batch):
		data = batch * batch_size
		length = int(50 * data / total_data)
		bar = length * '=' + '>' + (49 - length) * ' '
		print('epoch {}/{} ({}/{}) [{}]'.format(epoch , total_epoch , data , total_data , bar) , end = '\r')
	else:
		data = total_data
		bar = 50 * '='
		print('epoch {}/{} ({}/{}) [{}] ({}s) train RMSE : {:.5f} , validation RMSE : {:.5f}'.format(epoch , total_epoch , data , total_data , bar , total_time , train_RMSE , validation_RMSE))
	return