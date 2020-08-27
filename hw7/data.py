import os
import pandas as pd
import numpy as np

def load_data(train_directory , validation_directory , test_directory):
	if (train_directory is not None):
		train_x = list()
		train_y = list()
		for image_name in os.listdir(train_directory):
			train_x.append(os.path.join(train_directory , image_name))
			train_y.append(int(image_name.split('_')[0]))

		validation_x = list()
		validation_y = list()
		for image_name in os.listdir(validation_directory):
			validation_x.append(os.path.join(validation_directory , image_name))
			validation_y.append(int(image_name.split('_')[0]))

		return (train_x , train_y , validation_x , validation_y)
	else:
		test_x = list()
		for image_name in sorted(os.listdir(test_directory)):
			test_x.append(os.path.join(test_directory , image_name))
		return test_x

def dump(predict , output_file):
	number_of_data = predict.shape[0]
	df = pd.DataFrame({'Id' : np.arange(number_of_data) , 'label' : predict})
	df.to_csv(output_file , index = False)
	return