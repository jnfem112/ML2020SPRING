import os
import pandas as pd
from PIL import Image
import cv2

def load_data(input_directory , output_directory = None):
	original_image = list()
	original_label = list()
	new_image = list()
	df = pd.read_csv(os.path.join(input_directory , 'labels.csv'))
	for i in range(df.shape[0]):
		original_image.append(Image.open(os.path.join(input_directory , 'images/{:03d}.png'.format(df['ImgId'][i]))))
		original_label.append(df['TrueLabel'][i])
		if (output_directory is not None):
			new_image.append(Image.open(os.path.join(output_directory , '{:03d}.png'.format(df['ImgId'][i]))))
	return (original_image , original_label) if (output_directory is None) else (original_image , original_label , new_image)

def save_image(image , output_directory):
	for i in range(len(image)):
		image[i] = cv2.cvtColor(image[i] , cv2.COLOR_RGB2BGR)
		cv2.imwrite(os.path.join(output_directory , '{:03d}.png'.format(i)) , image[i])
	return