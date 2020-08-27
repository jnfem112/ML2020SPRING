import numpy as np
from time import time
from utils import my_argparse , print_progress
from data import load_data
from model import save_model

def RMSE(x , y , weight , bias):
	return np.sqrt(np.mean(((x @ weight + bias) - y.reshape(-1 , 1))**2))

# Implement Adam for linear regression.
def linear_regression(train_x , train_y , validation_x , validation_y , args):
	# Initialization.
	(number_of_data , dimension) = train_x.shape
	weight = np.full(dimension , 0.1).reshape(-1 , 1)
	bias = 0.1
	
	# Hyper-parameter.
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	lambd = args.lambd
	beta_1 = np.full(dimension , 0.9).reshape(-1 , 1)
	beta_2 = np.full(dimension , 0.99).reshape(-1 , 1)
	m_t = np.full(dimension , 0).reshape(-1 , 1)
	v_t = np.full(dimension , 0).reshape(-1 , 1)
	m_t_b = 0
	v_t_b = 0
	t = 0
	epsilon = 1e-8
	epoch = args.epoch
	
	for i in range(epoch):
		start = time()

		# Shuffle training data.
		index = np.arange(number_of_data)
		np.random.shuffle(index)
		train_x = train_x[index]
		train_y = train_y[index]

		for j in range(number_of_data // batch_size):
			t += 1
			x_batch = train_x[j * batch_size : (j + 1) * batch_size]
			y_batch = train_y[j * batch_size : (j + 1) * batch_size].reshape(-1 , 1)
			loss = y_batch - np.dot(x_batch , weight) - bias
			
			# Calculate gradient.
			g_t = -2 * np.dot(x_batch.transpose() , loss) + 2 * lambd * np.sum(weight)
			g_t_b = 2 * loss.sum(axis = 0)
			m_t = beta_1 * m_t + (1 - beta_1) * g_t 
			v_t = beta_2 * v_t + (1 - beta_2) * np.multiply(g_t , g_t)
			m_cap = m_t / (1 - beta_1**t)
			v_cap = v_t / (1 - beta_2**t)
			m_t_b = 0.9 * m_t_b + (1 - 0.9) * g_t_b
			v_t_b = 0.99 * v_t_b + (1 - 0.99) * (g_t_b * g_t_b) 
			m_cap_b = m_t_b / (1 - 0.9**t)
			v_cap_b = v_t_b / (1 - 0.99**t)
			w_0 = np.copy(weight)
			
			# Update weight and bias.
			weight -= ((learning_rate * m_cap) / (np.sqrt(v_cap) + epsilon)).reshape(-1 , 1)
			bias -= (learning_rate * m_cap_b) / (np.sqrt(v_cap_b) + epsilon)

			# Calculate loss.
			train_RMSE = RMSE(train_x , train_y , weight , bias)
			validation_RMSE = RMSE(validation_x , validation_y , weight , bias)
			end = time()
			print_progress(i + 1 , epoch , number_of_data , batch_size , j + 1 , number_of_data // batch_size , int(end - start) , train_RMSE , validation_RMSE)

	return (weight , bias)

def main(args):
	(train_x , train_y , validation_x , validation_y) = load_data(args.train_data)
	(weight , bias) = linear_regression(train_x , train_y , validation_x , validation_y , args)
	save_model(weight , bias)
	return

if (__name__ == '__main__'):
	args = my_argparse()
	main(args)