import pandas as pd

def dump(predict , output_file):
	number_of_data = predict.shape[0]
	df = pd.DataFrame({'id' : np.arange(number_of_data) , 'label' : predict})
	df.to_csv(output_file , index = False)
	return