import dgl
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import re



'''
Build a heterograph structure from given canonical edge type tuples and 
corresponding edge indices.

params:
	-cet_tuples: a list containing all canonical edge type tuples.
	-edge_indices_dir: directory containing all edge indices.

return:
	G: a dgl heterograph strcutre.
'''


def build_heterograph_struct(cet_tuples, edge_indices_dir):

	print('Building strucutre of the heterograph')

	#a dictionary specifying the structural information of the heterograph to
	#be built
	struct_dict = {}

	for idx, cet_tup in enumerate(cet_tuples):

		relation = cet_tup[1]

		if(relation.endswith('_rev')):
			continue

		cet_str = '_'.join(cet_tup)
	       
		edge_indices = np.load(edge_indices_dir+cet_str+'_Edge_Indices.npy')

		struct_dict[cet_tup] = (edge_indices[0],edge_indices[1])


		if(re.match('self_loop\d+', relation)):
			#self loop edge types have no reverse edge
			continue

		#reverse edges
		cet_rev_tup = cet_tuples[idx+1]

		struct_dict[cet_rev_tup] = (edge_indices[1], edge_indices[0])


	G = dgl.heterograph(struct_dict)
	
	print('Finish building strucutre of the heterograph')
	print('\n')

	return G




def load_initial_embeddings(node_name_list, pre_trained_flag_list,
 textual_feature_flag_list, pre_trained_embs_dir, 
 compressed_cf_embs_dir, ntf_embs_dir, compress_size):

	print('Loading initial embeddings')

	#a dictionary whose keys are names of nodes and edges and the values 
	#are corresponding initial embeddings
	init_embs = {}
	for name, p_flag,  t_flag in zip(node_name_list, 
		pre_trained_flag_list, textual_feature_flag_list):
		print('Loading ', name)
		print('\n')

		#Get original node/edge name (e.g., Sample_Clean_Questions -> Questions)
		ori_name = name.split('Clean_')[-1]

		if(p_flag):
			init_embs[ori_name] = torch.load(pre_trained_embs_dir+name+\
				'_Pre_Trained_Embeddings.pt').to(torch.device('cpu'))


		elif(t_flag):
			init_embs[ori_name] = torch.load(compressed_cf_embs_dir \
				+name+'_Compressed_CF_Emb_Dim_{}.pt'.format(\
					compress_size)).to(torch.device('cpu'))


		else:
			init_embs[ori_name] = torch.as_tensor(\
				np.load(ntf_embs_dir+name+\
					'_Non_Textual_Features_Embeddings.npy'),
				dtype=torch.float, device=torch.device('cpu'))
		
	print('Finish loading initial embeddings')
	print('\n')

	return init_embs






def load_data(task_name_list, task_dir_list, task_type_list, mode):

	id_list = []
	target_list = []

	if(mode not in {'train', 'dev', 'test'}):
		print('Invalid mode: {}. Mode can only be train, dev or test'.format(mode))
		sys.exit()


	for task_name, task_dir, task_type in zip(task_name_list, task_dir_list,
	 task_type_list):


		task_sub_dir = task_dir + mode +'/'


		id_list.append(json.load(open(task_sub_dir + mode \
			+ '_new_id_list.json') ) )

		if(task_type == 'ranking'): #ranking task
			target_list.append(json.load(open(task_sub_dir + mode \
			+ '_target_id_list.json')))

		else: #linke prediction or classification task

			target_list.append(json.load(open(task_sub_dir + mode \
			+ '_label_list.json')))


	return id_list, target_list




def compute_classification_task_loss(ids, node_logits, 
	input_type_list,task_specific_output_layer, 
	targets):
	

	input_logits = node_logits[input_type_list[0]][ids]

	preds = task_specific_output_layer(\
		input_logits)

	task_loss = F.cross_entropy(preds, targets.long())

	return task_loss.view(1)
	


def compute_link_prediction_task_loss(ids, node_logits,
 input_type_list, task_specific_output_layer, targets, device, 
 pos_weight):
	
	src_node_logits = node_logits[input_type_list[0]]\
	[ids[:,0]]

	dst_node_logits = node_logits[input_type_list[-1]]\
	[ids[:,1]]


	preds = torch.mv(src_node_logits * dst_node_logits, 
		task_specific_output_layer)	

	
	task_loss = F.binary_cross_entropy_with_logits(preds, targets, 
		pos_weight = torch.tensor([pos_weight], dtype=torch.float, 
			device=device))

	return task_loss.view(1)



def compute_ranking_task_loss(ids,  node_logits, 
	input_type_list, task_specific_output_layer, targets, device):

	neg_log_list = []
	targets = torch.tensor(targets, device =device).long()

	for sample, target in zip(ids, targets):
		q_ids = torch.tensor([sample[0]]*len(sample[1]), device =device).long()
		q_logits = node_logits[input_type_list[0]][q_ids]
	
		a_ids = torch.tensor(sample[1], device =device).long()
		a_logits = node_logits[input_type_list[-1]][a_ids]

		concate_logits = torch.cat((q_logits, a_logits),dim=1)

		scores = F.leaky_relu(task_specific_output_layer(concate_logits))

		target_score = scores[target]

		idx = torch.arange(start=0, end =concate_logits.shape[0], dtype=torch.long)

		diff_scores = target_score - (scores[idx!=target])

		log_sigmoid = torch.mean(F.logsigmoid(diff_scores))


		neg_log_list.append((-1)*log_sigmoid.view(1))


	task_loss = torch.mean(torch.cat(neg_log_list, 0))

	return task_loss.view(1)
	


def compute_constraint_1_loss(constraint_1_list, 
	ans_score_clf_output_layer, ans_rec_output_layer, 
	 node_logits, device):
	
	

	neg_log_list = []

	for q, a_list in constraint_1_list:
		
		a_ids = torch.tensor(a_list, device =device).long()
		a_logits = node_logits['Answers'][a_ids]

		a_scores = ans_score_clf_output_layer(a_logits)

		max_score = torch.max(a_scores)

		max_score_idx_list = (a_scores==max_score).nonzero()[:, 0]

		q_ids = torch.tensor([q]*len(max_score_idx_list), device =device).long()
		q_logits = node_logits['Questions'][q_ids]

		concate_logits = torch.cat((q_logits, a_logits[max_score_idx_list]),dim=1)

		scores = F.leaky_relu(ans_rec_output_layer(concate_logits))

		log_sigmoid_mean_score = torch.mean(F.logsigmoid(scores))

		neg_log_list.append((-1)*log_sigmoid_mean_score.view(1))



	constraint_1_loss = torch.mean(torch.cat(neg_log_list, 0))

	return constraint_1_loss.view(1)


def compute_constraint_2_loss(G, constraint_2_list, 
	user_reputation_clf_output_layer, ans_rec_output_layer, 
	node_logits, device):


	neg_log_list = []

	for q, a_list in constraint_2_list:
	

		u_ids = torch.tensor([G.predecessors(a, 'owner2_of').item()\
		 for a in a_list], device =device, dtype=torch.long)

		u_logits = node_logits['Users'][u_ids]

		u_reputations = user_reputation_clf_output_layer(u_logits)

		max_reputation = torch.max(u_reputations)

		max_reputation_idx_list = (u_reputations==max_reputation).\
		nonzero()[:, 0]

		q_ids = torch.tensor([q]*len(max_reputation_idx_list), device =device)\
		.long()

		q_logits = node_logits['Questions'][q_ids]

		a_ids = torch.tensor(a_list, device =device).long()

		a_logits = node_logits['Answers'][a_ids[max_reputation_idx_list]]

		concate_logits = torch.cat((q_logits, a_logits),dim=1)

		scores = F.leaky_relu(ans_rec_output_layer(concate_logits))

		log_sigmoid_mean_score = torch.mean(F.logsigmoid(scores))

		neg_log_list.append((-1)*log_sigmoid_mean_score.view(1))



	constraint_2_loss = torch.mean(torch.cat(neg_log_list, 0))

	return constraint_2_loss.view(1)


def evaluate_classification_task(ids, node_logits, 
	input_type_list, task_specific_output_layer, targets):

	input_logits = node_logits[input_type_list[0]][ids]

	preds = task_specific_output_layer(input_logits).argmax(dim=1).to(torch.\
		device('cpu')).numpy()


	targets = targets.to(torch.device('cpu')).numpy()

	score_dict = {}

	score_dict['Accuracy'] = accuracy_score(targets, preds)

	score_dict['Macro F1'] = f1_score(targets, preds, average = 'macro')



	return score_dict


def evaluate_link_prediction_task(ids, node_logits, 
	input_type_list, task_specific_output_layer, targets, 
	link_prediction_threshold, device):


	src_node_logits = node_logits[input_type_list[0]][ids[:,0]]

	dst_node_logits = node_logits[input_type_list[-1]][ids[:,1]]


	preds = torch.mv(src_node_logits * dst_node_logits, 
		task_specific_output_layer).data	


	score_dict = {}

	targets = targets.to(torch.device('cpu')).numpy()

	preds = nn.Sigmoid()(preds).to(torch.device('cpu')).numpy()

	preds = (preds >= link_prediction_threshold).astype(int)

	score_dict['Accuracy'] = accuracy_score(targets, preds)

	score_dict['F1'] = f1_score(targets, preds)


	return score_dict



def evaluate_ranking_task(ids, node_logits, 
	input_type_list, task_specific_output_layer, targets, device):

	hit_count = 0
	ndcg_sum = 0

	targets = torch.tensor(targets, device =device).long()

	for sample, target in zip(ids, targets):

		q_ids = torch.tensor([sample[0]]*len(sample[1]), device =device).long()
		q_logits = node_logits[input_type_list[0]][q_ids]

		a_ids = torch.tensor(sample[-1], device =device).long()
		a_logits = node_logits[input_type_list[1]][a_ids]

		concate_logits = torch.cat((q_logits, a_logits),dim=1)

		scores = F.leaky_relu(task_specific_output_layer(concate_logits)).squeeze()


		top_3_id_list = torch.topk(scores, k=3)[1]

		if(target in top_3_id_list):
			hit_count+=1

			idx = (top_3_id_list == target).nonzero().item()

			ndcg_sum+= math.log(2) / math.log(idx+2)


	score_dict = {}

	score_dict['Hit Ratio'] = hit_count/len(ids)

	score_dict['NDCG'] = ndcg_sum/len(ids)



	return score_dict
	

def compress_embs(init_embs, compress_size, device):

	pca = PCA(n_components=compress_size)

	comp_embs = {name: nn.Parameter(torch.as_tensor(\
		pca.fit_transform(emb), dtype=torch.float, device=device)) \
	for name, emb in init_embs.items()}

	return comp_embs


def identity(x):
	return x

