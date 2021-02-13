import torch.nn as nn

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder , self).__init__()

		self.encoder = nn.Sequential(
				nn.Conv2d(3 , 64 , kernel_size = 3 , stride = 1 , padding = 1) ,
				nn.ReLU() ,
				nn.MaxPool2d(2) ,
				nn.Conv2d(64 , 128 , kernel_size = 3 , stride = 1 , padding = 1) ,
				nn.ReLU() ,
				nn.MaxPool2d(2) ,
				nn.Conv2d(128 , 256 , kernel_size = 3 , stride = 1 , padding = 1) ,
				nn.ReLU() ,
				nn.MaxPool2d(2)
			)
		 
		self.decoder = nn.Sequential(
				nn.ConvTranspose2d(256 , 128 , kernel_size = 5 , stride = 1) ,
				nn.ReLU() ,
				nn.ConvTranspose2d(128 , 64 , kernel_size = 9 , stride = 1) ,
				nn.ReLU() ,
				nn.ConvTranspose2d(64 , 3 , kernel_size = 17 , stride = 1) ,
				nn.Tanh()
			)

		return

	def forward(self , x):
		encode_x = self.encoder(x)
		decode_x  = self.decoder(encode_x)
		return (encode_x , decode_x)