import os
import pandas as pd
import numpy as np
import cv2

def load_data(directory , return_label):
	x = list()
	y = list()
	for image_name in os.listdir(directory):
		image = cv2.imread(os.path.join(directory , image_name) , cv2.IMREAD_COLOR)
		image = cv2.resize(image , (128 , 128) , interpolation = cv2.INTER_CUBIC)
		x.append(image)
		if (return_label):
			label = int(image_name.split('_')[0])
			y.append(label)
	if (return_label):
		return (np.array(x) , np.array(y))
	else:
		return np.array(x)

def load_dataset(train_directory , validation_directory , test_directory):
	if (train_directory is not None):
		(train_x , train_y) = load_data(train_directory , True)
		(validation_x , validation_y) = load_data(validation_directory , True)
		return (train_x , train_y , validation_x , validation_y)
	else:
		test_x load_data(test_directory , False)
		return test_x

def dump(predict , output_file):
	number_of_data = predict.shape[0]
	df = pd.DataFrame({'Id' : np.arange(number_of_data) , 'Category' : predict})
	df.to_csv(output_file , index = False)
	return