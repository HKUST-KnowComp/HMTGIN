import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from util import identity, load_initial_embeddings
import re
import math

class MLP(nn.Module):
	def __init__(self, num_layers, input_dim, hidden_dim, output_dim, 
		activation_func):
		super(MLP, self).__init__()

		self.num_layers = num_layers
		self.activation_func = activation_func


		if (num_layers == 1):
			self.linear = nn.Linear(input_dim, output_dim)

		else:
			#Multi-layer model
			
			self.linears = torch.nn.ModuleList()

			self.linears.append(nn.Linear(input_dim, hidden_dim))
			for layer in range(num_layers - 2):
				self.linears.append(nn.Linear(hidden_dim, hidden_dim))
			self.linears.append(nn.Linear(hidden_dim, output_dim))



	def forward(self, x):

		if(self.num_layers == 1):
			h = self.linear(x)

		else:
			h = x

			for layer in range(self.num_layers - 1):
				h = self.activation_func(self.linears[layer](h))
				

			h = self.linears[self.num_layers - 1](h)


		return h







class HeteroRGINLayer(nn.Module):

	def __init__(self, in_size, out_size, etypes, activation_name,
		activation_func, dropout_rate, device, mlp, batch_norm,
		epsilon, bias_flag):

		super(HeteroRGINLayer, self).__init__()

		self.in_size = in_size
		self.out_size = out_size
		self.activation_name = activation_name
		self.activation_func = activation_func
		self.device = device
		self.epsilon = epsilon
		self.bias_flag = bias_flag	
		self.mlp = mlp
		self.batch_norm = batch_norm

	

		self.weight = nn.ParameterDict({etype: nn.Parameter(torch.Tensor(in_size, 
			out_size)) for etype in etypes})

		for etype in self.weight:
			 nn.init.xavier_uniform_(self.weight[etype], 
				gain=nn.init.calculate_gain(activation_name))


		if(self.bias_flag):
			self.bias = nn.Parameter(torch.Tensor(out_size))

			uni_bound = math.sqrt(1/(in_size))

			nn.init.uniform_(self.bias, a=-uni_bound, b=uni_bound)

		self.dropout = nn.Dropout(dropout_rate)




	def forward(self, G, feat_dict):
		
		
	
		func_dict = {}

		for srctype, etype, dsttype in G.canonical_etypes:

			
			# Compute message
			nodes = feat_dict[srctype]


			if(re.match('self_loop\d+', etype)):
				nodes = (1+self.epsilon)*nodes

		
			msg = torch.mm(nodes, self.weight[etype]) 


			# Save it in graph for message passing

			G.nodes[srctype].data['msg_{}'.format(etype)] \
			= msg

			# Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.

			func_dict[etype] = (fn.copy_u('msg_{}'.format(etype),
			 'm'), fn.sum('m', 'h'))

			if(self.device == torch.device('cuda')):
				torch.cuda.empty_cache()



		# Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
		

		G.multi_update_all(func_dict, 'sum')

		if(self.device == torch.device('cuda')):
			torch.cuda.empty_cache()

        # return the updated node feature dictionary


		h_dict = { ntype : self.activation_func(G.nodes[ntype].data['h'] + \
			self.bias if self.bias_flag else G.nodes[ntype].data['h']) \
		for ntype in G.ntypes}

		if(self.mlp is not None):

			h_dict = {ntype: self.mlp(h) for ntype, h in h_dict.items()}
		
		if(self.batch_norm is not None):
			h_dict = {ntype: self.batch_norm(h) for ntype, h in h_dict.items()}

		h_dict = {ntype: self.dropout(self.activation_func(h)) for ntype, h in \
		h_dict.items()}


		return h_dict



class HeteroRGIN(nn.Module):
	
	def __init__(self, etypes, in_size, hidden_size_list, out_size, 
		init_embs_dir, num_hidden_layer,  num_mlp_layers_list, batch_norm_flag_list, 
		activation_name_list, dropout_rate_list, device, epsilon_list, 
		epsilon_trainable_list, bias_flag_list):

		super(HeteroRGIN, self).__init__()

		
		self.activation_name_list = activation_name_list

		self.activation_func_dict = {'relu': F.relu, 
		'leaky_relu':F.leaky_relu, 'linear': identity }

		self.device = device
		self.mlps = torch.nn.ModuleList()
		self.batch_norms = torch.nn.ModuleList()

		mlp_dims = [dim for dim in hidden_size_list ] + [out_size]

		for idx, num_mlp_layer in enumerate(num_mlp_layers_list):

			if(num_mlp_layer > 0):
				self.mlps.append(MLP(num_mlp_layer, mlp_dims[idx], 
					mlp_dims[idx], mlp_dims[idx],
					self.activation_func_dict[activation_name_list[idx]]))

			else:
				self.mlps.append(None)

			if(batch_norm_flag_list[idx]):
				self.batch_norms.append(nn.BatchNorm1d(mlp_dims[idx]))
			else:
				self.batch_norms.append(None)
			

		epsilon_trainable_list = [True if flag else False for flag in \
		epsilon_trainable_list]

		


		self.layers = nn.ModuleList([HeteroRGINLayer(\
			in_size, hidden_size_list[0],  etypes, activation_name_list[0], 
			self.activation_func_dict[activation_name_list[0]],
			dropout_rate_list[0], self.device, self.mlps[0], self.batch_norms[0],
			torch.tensor(epsilon_list[0], dtype=torch.float, device= self.device, 
				requires_grad=epsilon_trainable_list[0]), bias_flag_list[0])])

		for i in range(num_hidden_layer):

			self.layers.append(HeteroRGINLayer(\
			hidden_size_list[i], hidden_size_list[i+1],  
			etypes,   activation_name_list[i+1], 
			self.activation_func_dict[activation_name_list[i+1]],
			dropout_rate_list[i+1], self.device, self.mlps[i+1], self.batch_norms[i+1],
			torch.tensor(epsilon_list[i+1], dtype=torch.float, device= self.device, 
				requires_grad=epsilon_trainable_list[i+1]), bias_flag_list[i+1]))

		#No non-linear activation for the output layer. 
		#Use identity function as activation.

		self.layers.append(HeteroRGINLayer(\
			hidden_size_list[-1], out_size,  
			etypes,  activation_name_list[-1], 
			self.activation_func_dict[activation_name_list[-1]],
			dropout_rate_list[-1], self.device, self.mlps[-1],  self.batch_norms[-1],
			torch.tensor(epsilon_list[-1], dtype=torch.float, device= self.device, 
				requires_grad=epsilon_trainable_list[-1]), bias_flag_list[-1]))

	def forward(self, G, init_embeds):

		h_dict = {node_name: embeds.to(self.device) for node_name,
			embeds in init_embeds.items()}


		for layer in self.layers:

			h_dict = layer(G, h_dict)

			if(self.device == torch.device('cuda')):
				torch.cuda.empty_cache()


		# get node logits 

		return h_dict
