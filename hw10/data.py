import pandas as pd
import numpy as np

def load_data(path):
	data = np.load(path , allow_pickle = True)
	return data

def dump(score , output_file):
	number_of_data = score.shape[0]
	df = pd.DataFrame({'id' : np.arange(1 , number_of_data + 1) , 'anomaly' : score})
	df.to_csv(output_file , index = False)
	return