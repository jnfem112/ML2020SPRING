import random as rd
import torch
import torch.nn as nn
import torch.nn.functional as F

class Node():
	def __init__(self , predict , hidden_state , outputs , probability):
		self.predict = predict
		self.hidden_state = hidden_state
		self.outputs = outputs
		self.probability = probability
		return

	def copy(self):
		return Node(self.predict.copy() , self.hidden_state , self.outputs.copy() , self.probability)

	def __lt__(self , another):
		return self.probability < another.probability

class Attention(nn.Module):
	def __init__(self , hidden_dim):
		super(Attention , self).__init__()
		self.hidden_dim = hidden_dim

		self.linear = nn.Sequential(
				nn.Linear(2 * hidden_dim , hidden_dim , bias = True) ,
				nn.Tanh() ,
				nn.Linear(hidden_dim , 1 , bias = False)
			)

		return

	def forward(self , hidden_state , encoder_output):
		hidden_state = hidden_state[-1].unsqueeze(dim = 1).repeat(1 , encoder_output.size(dim = 1) , 1)
		score = self.linear(torch.cat((hidden_state , encoder_output) , dim = 2)).squeeze(dim = 2)
		weight = F.softmax(score , dim = 1).unsqueeze(dim = 1)
		attention = torch.bmm(weight , encoder_output)
		return attention

class Encoder(nn.Module):
	def __init__(self , num_embeddings , embedding_dim = 256 , hidden_dim = 512 , num_layers = 3 , dropout = 0.5):
		super(Encoder , self).__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(num_embeddings , embedding_dim)
		self.dropout = nn.Dropout(dropout)
		self.recurrent = nn.GRU(embedding_dim , hidden_dim , num_layers = num_layers , bias = True , batch_first = True , dropout = dropout , bidirectional = True)

		return

	def forward(self , input):
		input = self.embedding(input)
		input = self.dropout(input)
		(output , hidden_state) = self.recurrent(input)
		return (output , hidden_state)

class Decoder(nn.Module):
	def __init__(self , num_embeddings , embedding_dim = 256 , hidden_dim = 1024 , num_layers = 3 , dropout = 0.5 , apply_attention = True):
		super(Decoder , self).__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.apply_attention = apply_attention
		self.attention = Attention(hidden_dim)

		self.embedding = nn.Embedding(num_embeddings , embedding_dim)
		self.dropout = nn.Dropout(dropout)
		self.recurrent = nn.GRU(embedding_dim + apply_attention * hidden_dim , hidden_dim , num_layers = num_layers , bias = True , batch_first = True , dropout = dropout , bidirectional = False)
		self.linear = nn.Sequential(
				nn.Linear(hidden_dim + apply_attention * (embedding_dim + hidden_dim) , 2 * hidden_dim) ,
				nn.Linear(2 * hidden_dim , 4 * hidden_dim) ,
				nn.Linear(4 * hidden_dim , num_embeddings)
			)

		return

	def forward(self , input , hidden_state , encoder_output):
		input = input.unsqueeze(dim = 1)
		input = self.embedding(input)
		input = self.dropout(input)
		if (self.apply_attention):
			attention = self.attention(hidden_state , encoder_output)
			(output , hidden_state) = self.recurrent(torch.cat((input , attention) , dim = 2) , hidden_state)
			output = self.linear(torch.cat((input , attention , output) , dim = 2).squeeze(dim = 1))
		else:
			(output , hidden_state) = self.recurrent(input , hidden_state)
			output = output.squeeze(dim = 1)
			output = self.linear(output)
		return (output , hidden_state)

class Seq2Seq(nn.Module):
	def __init__(self , encoder , decoder):
		super(Seq2Seq , self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		return

	def forward(self , input , target , teacher_forcing_ratio):
		(encoder_output , hidden_state) = self.encoder(input)
		hidden_state = hidden_state.view(self.encoder.num_layers , 2 , input.size(dim = 0) , -1)
		hidden_state = torch.cat((hidden_state[ : , -2 , : , : ] , hidden_state[ : , -1 , : , : ]) , dim = 2)

		input = target[ : , 0]
		outputs = list()
		predict = list()
		for i in range(1 , target.size(dim = 1)):
			(output , hidden_state) = self.decoder(input , hidden_state , encoder_output)
			index = output.argmax(dim = 1)
			outputs.append(output.unsqueeze(dim = 1))
			predict.append(index.unsqueeze(dim = 1))
			teacher_forcing = rd.random() <= teacher_forcing_ratio
			input = target[ : , i] if (teacher_forcing and i < target.size(dim = 1)) else index
		outputs = torch.cat(outputs , dim = 1)
		predict = torch.cat(predict , dim = 1)
		return (outputs , predict)

	def inference(self , input , target , beam_size = 1):
		(encoder_output , hidden_state) = self.encoder(input)
		hidden_state = hidden_state.view(self.encoder.num_layers , 2 , input.size(dim = 0) , -1)
		hidden_state = torch.cat((hidden_state[ : , -2 , : , : ] , hidden_state[ : , -1 , : , : ]) , dim = 2)

		node = Node(list() , hidden_state , list() , 1)
		node_list = [node]
		for i in range(1 , target.size(dim = 1)):
			new_node_list = list()
			for node in node_list:
				input = target[ : , 0] if (i == 1) else node.predict[-1].squeeze(dim = 1)
				hidden_state = node.hidden_state
				(output , hidden_state) = self.decoder(input , hidden_state , encoder_output)
				(value , index) = F.softmax(output / 5 , dim = 1).topk(beam_size , dim = 1)
				for j in range(beam_size):
					new_node = node.copy()
					new_node.predict.append(index[ : , j].unsqueeze(dim = 1))
					new_node.hidden_state = hidden_state
					new_node.outputs.append(output.unsqueeze(dim = 1))
					new_node.probability *= value[ : , j].item()
					new_node_list.append(new_node)
			new_node_list.sort(reverse = True)
			node_list = new_node_list[ : beam_size]
		outputs = torch.cat(node_list[0].outputs , dim = 1)
		predict = torch.cat(node_list[0].predict , dim = 1)
		return (outputs , predict)