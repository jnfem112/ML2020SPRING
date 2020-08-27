import torch 
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self , embedding , padding_index):
		super(RNN , self).__init__()

		self.embedding = nn.Embedding(embedding.size(dim = 0) , embedding.size(dim = 1) , padding_idx = padding_index)
		self.embedding.weight = nn.Parameter(embedding)
		self.embedding.weight.requires_grad = False

		self.recurrent = nn.LSTM(embedding.size(dim = 1) , 150 , batch_first = True , bias = True , num_layers = 2 , dropout = 0.3 , bidirectional = True)
		
		self.linear = nn.Sequential(
			nn.Dropout(0.5) ,
			nn.Linear(900 , 1 , bias = True) ,
			nn.Sigmoid()
		)

		return

	def forward(self , x):
		x = self.embedding(x)
		(x , _) = self.recurrent(x)
		x = torch.cat([x.min(dim = 1).values , x.max(dim = 1).values , x.mean(dim = 1)] , dim = 1)
		x = self.linear(x)
		return x