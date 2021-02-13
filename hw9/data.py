import pandas as pd
import numpy as np

def load_data(path):
	data = np.load(path)
	return data

def dump(predict , output_file):
	number_of_data = predict.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : predict})
	df.to_csv(output_file , index = False)
	return