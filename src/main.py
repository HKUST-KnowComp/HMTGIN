import argparse
import dgl
from util import *
from model import HeteroRGIN
import torch
torch.manual_seed(0)
import torch.nn as nn
import numpy as np
np.random.seed(0)
import math
import json
import pandas as pd
import sys
from torch.utils.tensorboard import SummaryWriter



def main():

	parser = argparse.ArgumentParser(description='Main program of SO_MTL model.')

	parser.add_argument('--cet_list_path', type = str, required=True,
		help='Path to the list containing all canonical edge type strings.')

	parser.add_argument('--edge_indices_dir', type = str, required=True,
		help='Directory containing all edge indices.')

	parser.add_argument('--node_name_list', type = str,
	 required=True, nargs='+',	help='A list containing names of nodes and edges')

	parser.add_argument('--textual_feature_flag_list', type = int, required=True, nargs ='+',
		help='A list containing flags specifying whether corresponding nodes have textual features. 1 means True, 0 means False.')

	parser.add_argument('--compressed_cf_embs_dir', type = str, 
		help='Directory containing all compressed combined features embeddings.')

	parser.add_argument('--ntf_embs_dir', type = str, 
			help='Directory containing all non-textual features embeddings.')

	parser.add_argument('--task_name_list', type = str, required=True, 
		nargs='+',	help="A list containing names of all tasks.")

	
	parser.add_argument('--task_dir_list', type = str, required=True, 
		nargs='+',	help="A list containing directories of all tasks' datasets. The order follows that of task_name_list.")


	parser.add_argument('--task_type_list', type = str, required=True, 
		choices= ['link_prediction', 'classification', 'ranking'],	nargs='+',	
		help="A list containing types of all tasks. Currently only link_prediction, classification and ranking tasks are supported. The order follows that of task_name_list.")

	parser.add_argument('--constraint_1_coefficient', type = float, default = 0,
		help="Coefficient of constraint_1 loss in the total loss")

	parser.add_argument('--constraint_2_coefficient', type = float, default = 0,
		help="Coefficient of constraint_2 loss in the total loss")

	parser.add_argument('--patience', type=int, default=3,
		help="Stop training model if the number of epochs where the model's performance on validation set does not improve reaches patience.")

	parser.add_argument('--num_hidden_layer', type=int, default=0, required=True,
		help="Number of hidden layers of the MTL model.")

	parser.add_argument('--hidden_size_list', type=int, nargs='+', 
		required=True,	help="A list containing sizes of all hidden layer(s) for all canonical edge types.")

	parser.add_argument('--residual_size', type=int, default=16,
		help="Residual sizes of all canonical edge types.")

	parser.add_argument('--compress_size', type=int, default=16,
		help="Node/edge embedding size after compression.")

	parser.add_argument('--max_num_epoch', type=int, default=20,
		help="Maximum number of epochs")

	parser.add_argument('--constraint_1_list_path', type = str, 
		 help='Path to the question answer list for computing the constraint 1 loss.')

	parser.add_argument('--constraint_2_list_path', type = str, 
		help='Path to the question answer list for computing the constraint 2 loss.')

	parser.add_argument('--mode',choices = ['train', 'test'],
		required=True, help='Mode of the execution')

	parser.add_argument('--checkpoint_path', type = str, required=True,
		help='Path to checkpoint')

	parser.add_argument('--link_prediction_threshold', type=float,
	 default=0.5, help="Classification threshold for link prediction tasks.")

	
	parser.add_argument('--num_mlp_layers_list', type=int, nargs ='+', required =True,
	help="A list containing numbers of MLP layers in all layers.")


	parser.add_argument('--dropout_rate_list', type=float, required=True,
		nargs='+', 
		help="A list containing probabilities of an element to be zeroed in all layers of MTL model.")

	parser.add_argument('--activation_name_list', type=str, nargs='+', 
		required=True,	choices=['relu', 'leaky_relu',
		 'linear'], help="A list containing the name of activation functions of all layers of the MTL model.")

	parser.add_argument('--init_lr', type=float,
	 default=0.01, help="Initial learning rate for optimizer.")

	parser.add_argument('--lr_reduce_step', type=int,  default=50,
		help="Decrease learning rate in every lr_reduce_step epoch.")

	parser.add_argument('--lr_reduce_factor', type=float,
	 default=0.5, help="Decrease lr by multiplying lr_reduce_factor.")

	parser.add_argument('--l2_penalty_coef', type=float,
	 default=0.01, help="l2 penalty coefficient for optimizer.")

	parser.add_argument('--num_class_list', type=int, nargs='+', 
		help="A list containing the number of classes for all classification tasks. The order follows that of classification tasks in task_name_list.")

	parser.add_argument('--use_gpu', action='store_true',
		help='Whether the computation should be carried out on gpu.')

	parser.add_argument('--resume_train', action='store_true',
		help='If true, then load the saved checkpoint and resume training. Otherwise, start training from scratch.')

	parser.add_argument('--train_dev_result_path', type = str, required=True,
		help='Path to the file saving the results on train and dev sets.')

	parser.add_argument('--test_result_path', type = str, required=True,
		help='Path to the file saving the results on test set.')

	parser.add_argument('--link_prediction_pos_weight', type=int,
	 default=1, help="Weight of positive samples in the loss functions of link prediction tasks.")


	parser.add_argument('--epsilon_list', type=float, nargs ='+', required = True,
	help="A list containing epsilon value for all layers.")

	parser.add_argument('--epsilon_trainable_list', type=int, nargs ='+', required = True,
	help="A list specifying whether the epsilons in corresponding layers are trainable or not. 1 means True, 0 means False.")

	parser.add_argument('--bias_flag_list', type=int, nargs ='+', required = True,
	help="A list specifying whether the corresponding layers have bias or not. 1 means True, 0 means False.")

	parser.add_argument('--batch_norm_flag_list', type=int, nargs ='+', required = True,
	help="A list specifying whether the corresponding layers have batch normalization or not. 1 means True, 0 means False.")


	parser.add_argument('--pre_trained_flag_list', type=int, nargs ='+', required = True,
	help="A list specifying whether the corresponding nodes would use pre-trained embeddings or not. 1 means True, 0 means False.")

	parser.add_argument('--pre_trained_emb_dir', type = str,
		help='Directory containing all pre-trained embeddings.')

	'''
	parser.add_argument('--initial_embeddings_trainable', action='store_true',
		help='Whether the inital embeddings are trainable or not.')
	'''
	parser.add_argument('--tensorboard_log_dir', type = str, required=True,
		help='Directory for tensorboard log.')

	parser.add_argument('--task_coef_list', type=float, 
		nargs='+', required=True,
		help="A list containing the coefficients for all tasks' loss functions in total loss function.")






	args = parser.parse_args()


	if(args.use_gpu and torch.cuda.is_available()):
		args.device = torch.device('cuda')

	else:
		args.device = torch.device('cpu')




	#a list containing input type lists of all tasks
	task_input_types = []

	for task_idx, task_name in enumerate(args.task_name_list):
			
		task_dir = args.task_dir_list[task_idx]\
		
		task_input_types.append(json.load(open(task_dir \
				+'Input_Type_List.json')))


	print('Building graph')

	cet_list = json.load(open(args.cet_list_path))
	cet_tuples = [tuple(cet) for cet in cet_list]
	del cet_list

	
	#build structure of the heterograph 
	G = build_heterograph_struct(cet_tuples, 
		args.edge_indices_dir)

	print('Finish building graph')
	print('\n')


	load_initial_embeddings_params = {
		'node_name_list': args.node_name_list, 
		'pre_trained_flag_list': args.pre_trained_flag_list,
		'textual_feature_flag_list': \
		args.textual_feature_flag_list, 
		'pre_trained_embs_dir': args.pre_trained_emb_dir, 
		'compressed_cf_embs_dir': args.compressed_cf_embs_dir, 
		'ntf_embs_dir': args.ntf_embs_dir, 
		'compress_size': args.compress_size, 
		}


	init_embeds = load_initial_embeddings\
	(**load_initial_embeddings_params)


	model = HeteroRGIN(G.etypes, args.compress_size, args.hidden_size_list, args.residual_size, 
		args.compressed_cf_embs_dir, args.num_hidden_layer, args.num_mlp_layers_list, 
		args.batch_norm_flag_list, args.activation_name_list, args.dropout_rate_list, 
		args.device, args.epsilon_list,	args.epsilon_trainable_list, args.bias_flag_list)
	

	model.to(device=args.device)

	if(args.mode == 'train'):

		print('Training:')
		print('\n')


		constraint_1_list = None

		if(args.constraint_1_list_path is not None):
			constraint_1_list = json.load(open(args.constraint_1_list_path))
		

		constraint_2_list = None

		if(args.constraint_2_list_path is not None):
			constraint_2_list = json.load(open(args.constraint_2_list_path))
		

		best_avg_dev_score = torch.tensor([-1], device=args.device)


		begin_epoch = 0


		#trainable parameters
		params = list(model.parameters())

		#a list containing task specific layers of all tasks
		task_specific_output_layer_list = []

		clf_task_idx = 0

		for task_name, task_dir, task_type in zip(args.task_name_list, 
			args.task_dir_list, args.task_type_list):

			layer=None

			out_size = args.residual_size + args.compress_size



			if(task_type == 'classification'): #classification task
				layer = nn.Linear(out_size, args.num_class_list[\
					clf_task_idx]).to(device=args.device)

				params+=list(layer.parameters())
				clf_task_idx +=1

			elif(task_type == 'link_prediction'): #link prediction task

				layer = torch.randn(out_size, dtype=torch.float, device=args.device,
					requires_grad=True)

				params.append(layer)

			else: #ranking task

				layer = nn.Linear(2*out_size, 1).to(device=args.device)
				params+=list(layer.parameters())



			task_specific_output_layer_list.append(layer)




		opt = torch.optim.Adam(params, lr=args.init_lr, 
			weight_decay=args.l2_penalty_coef)


		if(args.resume_train):
			#consecutive_no_improvement_count = -1
			print('Loading checkpoint')
			checkpoint = torch.load(args.checkpoint_path)
			begin_epoch = checkpoint['begin_epoch']
			model.load_state_dict(checkpoint['model_state_dict'])
			model.to(args.device)


			best_avg_dev_score = checkpoint['best_avg_dev_score']

			for idx, layer in \
			enumerate(checkpoint['task_specific_output_layer_list_checkpoint']):
				if(isinstance(task_specific_output_layer_list[idx], nn.Module)):
					task_specific_output_layer_list[idx].load_state_dict(checkpoint['task_specific_output_layer_list_checkpoint'][idx])
					task_specific_output_layer_list[idx].to(args.device)
				else:
					task_specific_output_layer_list[idx] = checkpoint['task_specific_output_layer_list_checkpoint'][idx].to(args.device)
			
			
			opt.load_state_dict(checkpoint['optimizer_state_dict'])
			print('Finish loading checkpint')



		#Get train and dev set 
		print('Loading training and developing data')

		train_id_list, train_target_list = load_data(\
			args.task_name_list, args.task_dir_list,
			 args.task_type_list, 'train')

		dev_id_list, dev_target_list = load_data(args.task_name_list, 
			args.task_dir_list, args.task_type_list, 'dev')

		print('Finish loading training and developing data')
		print('\n')



			

		model.train()

		#writer for tensorboard
		writer = SummaryWriter(args.tensorboard_log_dir)

		for epoch in range(args.max_num_epoch):

			total_epoch = begin_epoch + epoch

			print('Total epoch#: ', total_epoch, '\n')

			residual_node_embeds = model(G, init_embeds)
			
			if(args.device == torch.device('cuda')):
				torch.cuda.empty_cache()
			



			node_embeds = {node_name: torch.cat((\
					init_embeds[node_name].to(args.device), residual_node_embed), 
			dim=1) for node_name, residual_node_embed in \
			residual_node_embeds.items()}


			if(args.device == torch.device('cuda')):
				torch.cuda.empty_cache()




			
			train_task_loss_list = []

			all_score_dict = {}

			train_score_list = []
			dev_score_list = []

			for task_idx, task_name in enumerate(args.task_name_list):

				task_type = args.task_type_list[task_idx]
				input_type_list = task_input_types[task_idx]
				task_specific_output_layer = \
				task_specific_output_layer_list[task_idx]

				train_ids = train_id_list[task_idx]
				dev_ids = dev_id_list[task_idx]				
				train_targets = train_target_list[task_idx]
				dev_targets = dev_target_list[task_idx]

				if(task_type != 'ranking'):
					train_ids = torch.tensor(train_ids, 
						device=args.device, dtype=torch.long)
					dev_ids = torch.tensor(dev_ids, 
						device=args.device, dtype=torch.long)
					train_targets = torch.tensor(train_targets, 
						device=args.device, dtype=torch.float)
					dev_targets = torch.tensor(dev_targets, 
						device=args.device, dtype=torch.float)

		 

				if(task_type == 'classification'):#classification task

					task_specific_output_layer.train()

					train_task_loss = compute_classification_task_loss(\
						train_ids, node_embeds, input_type_list,
						task_specific_output_layer, train_targets)

					task_specific_output_layer.eval()
					

					train_score_dict = evaluate_classification_task(\
						train_ids, node_embeds, input_type_list,
						task_specific_output_layer, train_targets)

					dev_score_dict = evaluate_classification_task(\
						dev_ids, node_embeds, input_type_list,
						task_specific_output_layer, dev_targets)

					task_specific_output_layer.train()



				elif(task_type == 'link_prediction'):#link prediction task

					train_task_loss = compute_link_prediction_task_loss(\
						train_ids, node_embeds, input_type_list, 
						task_specific_output_layer, train_targets, 
						args.device, args.link_prediction_pos_weight)

					train_score_dict = evaluate_link_prediction_task(
						train_ids, node_embeds, input_type_list, 
						task_specific_output_layer, train_targets,
						args.link_prediction_threshold, args.device)

					dev_score_dict = evaluate_link_prediction_task(
						dev_ids, node_embeds, input_type_list, 
						task_specific_output_layer, dev_targets,
						args.link_prediction_threshold, args.device)




				else:#ranking task   

					task_specific_output_layer.train()
					train_task_loss = \
					compute_ranking_task_loss(\
						train_ids, node_embeds, 
						input_type_list, 
						task_specific_output_layer, 
						train_targets, args.device)

					task_specific_output_layer.eval()

					train_score_dict = evaluate_ranking_task(\
						train_ids, node_embeds, input_type_list, 
						task_specific_output_layer, train_targets, args.device)

					dev_score_dict = evaluate_ranking_task(\
						dev_ids, node_embeds, input_type_list, 
						task_specific_output_layer, dev_targets, args.device)

					task_specific_output_layer.train()



				train_task_loss_list.append(args.task_coef_list[task_idx] * \
					train_task_loss)

				task_train_score_list = []
				task_dev_score_list = []

				for metric in train_score_dict:
					key = task_name + ' ' + metric
					all_score_dict[key] = [train_score_dict[metric], 
					dev_score_dict[metric]]

					writer.add_scalar('{} train {}'.format(task_name, 
						metric), train_score_dict[metric], total_epoch)

					writer.add_scalar('{} dev {}'.format(task_name, 
						metric), dev_score_dict[metric], total_epoch)

					task_train_score_list.append(train_score_dict[metric])

					task_dev_score_list.append(dev_score_dict[metric])
					
				train_score_list+=task_train_score_list
				dev_score_list+=task_dev_score_list

				print('{} train loss: {}\n'.format(task_name, train_task_loss))

				writer.add_scalar('{} train loss'.format(task_name), 
					train_task_loss, total_epoch)



				writer.add_scalar('{} avg train score'.format(task_name), 
					sum(task_train_score_list)/len(task_train_score_list), 
					total_epoch)

				writer.add_scalar('{} avg dev score'.format(task_name), 
					sum(task_dev_score_list)/len(task_dev_score_list), 
					total_epoch)

		

				if(args.device == torch.device('cuda')):
					torch.cuda.empty_cache()




			if(len(args.task_name_list) > 1):
				
				constraint_1_loss = 0


				if(args.constraint_1_coefficient > 0):
					constraint_1_loss = compute_constraint_1_loss(constraint_1_list,
					 task_specific_output_layer_list[args.task_name_list.\
					 index('Answer_Score_Classification')], 
					 task_specific_output_layer_list[args.task_name_list.\
					 index('Answer_Recommendation')], node_embeds,
					  device= args.device)

					print('constraint_1_loss: {}\n'.format(constraint_1_loss))

					writer.add_scalar('constraint_1_loss', constraint_1_loss, 
						total_epoch)


				constraint_2_loss = 0

				if(args.constraint_2_coefficient > 0):
					constraint_2_loss = compute_constraint_2_loss(G, constraint_2_list,
					 task_specific_output_layer_list[args.task_name_list.\
					 index('User_Reputation_Classification')], 
					 task_specific_output_layer_list[args.task_name_list.\
					 index('Answer_Recommendation')], node_embeds, 
					 device= args.device)
				
					print('constraint_2_loss: {}\n'.format(constraint_2_loss))

					writer.add_scalar('constraint_2_loss', constraint_2_loss, 
						total_epoch)


				train_total_loss = torch.mean(torch.cat(\
					train_task_loss_list, 0)) + \
				args.constraint_1_coefficient * constraint_1_loss + \
				args.constraint_2_coefficient * constraint_2_loss


				

				
			else:
				train_total_loss = train_task_loss_list[0]
			


			if(args.device == torch.device('cuda')):
				torch.cuda.empty_cache()

			print('train_total_loss: {}\n'.format(train_total_loss))

			avg_train_score = sum(train_score_list)/len(train_score_list)

			avg_dev_score = sum(dev_score_list)/len(dev_score_list)
				

			writer.add_scalar('train total loss', train_total_loss, 
					total_epoch)

			writer.add_scalar('avg train score', avg_train_score, 
					total_epoch)

			writer.add_scalar('avg dev score', avg_dev_score, 
					total_epoch)



			if(best_avg_dev_score < avg_dev_score):
				best_avg_dev_score = avg_dev_score

				print('Saving checkpoint')
				checkpoint = {
					'begin_epoch': total_epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': opt.state_dict(),
					'train_total_loss': train_total_loss,
					'avg_train_score': avg_train_score,
					'best_avg_dev_score': best_avg_dev_score, 
					'residual_node_embeds': residual_node_embeds
					}

				task_specific_output_layer_list_checkpoint = []

				for output_layer in task_specific_output_layer_list:
					if(isinstance(output_layer, nn.Module)):
						task_specific_output_layer_list_checkpoint.append(output_layer.state_dict())
					else:
						task_specific_output_layer_list_checkpoint.append(output_layer)

				checkpoint['task_specific_output_layer_list_checkpoint'] = task_specific_output_layer_list_checkpoint

				torch.save(checkpoint, args.checkpoint_path)

				print('Finish saving checkpoint')
				print('\n')


				print('Saving all scores')

				all_score_df = pd.DataFrame.from_dict(all_score_dict, 
				orient='index', columns = ['Train', 'Dev'])

				all_score_df.to_csv(args.train_dev_result_path)
				
				print('Finish saving all scores\n')


			if(args.device == torch.device('cuda')):
				torch.cuda.empty_cache()

			if(total_epoch > 0 and total_epoch % args.lr_reduce_step == 0):
				print('reduce learning rate')
				for g in opt.param_groups:
					g['lr'] = g['lr'] * args.lr_reduce_factor


			opt.zero_grad()

	

			train_total_loss.backward()

			if(args.device == torch.device('cuda')):
				torch.cuda.empty_cache()

	


			opt.step()


			if(args.device == torch.device('cuda')):
				torch.cuda.empty_cache()





	else:
		print('Testing:')
		print('\n')

		print('Loading test data\n')

		test_id_list, test_target_list = load_data(\
			args.task_name_list, args.task_dir_list, 
			args.task_type_list, 'test')

		print('Finish loading test data')
		print('\n')

		print('Loading checkpoint')

		checkpoint = torch.load(args.checkpoint_path)



		print('Testing model on epoch # ', checkpoint['begin_epoch'])
		print('\n')



		task_specific_output_layer_list = []

		clf_task_idx = 0
		out_size = args.residual_size + args.compress_size

		for idx, task_type in enumerate(args.task_type_list):
			
			if(task_type == 'classification'): #classification task
				layer = nn.Linear(out_size, 
					args.num_class_list[clf_task_idx])

				layer.load_state_dict(checkpoint\
					['task_specific_output_layer_list_checkpoint'][idx])
				layer.eval()
				clf_task_idx +=1


			elif(task_type == 'link_prediction'): 

				layer = checkpoint['task_specific_output_layer_list_checkpoint'][idx]

			else: #ranking task
				layer = nn.Linear(2*out_size, 1).to(device=args.device)

				layer.load_state_dict(checkpoint\
					['task_specific_output_layer_list_checkpoint'][idx])
				layer.eval()

			layer = layer.to(args.device)

			task_specific_output_layer_list.append(layer)

	

		node_embeds = {node_name: torch.cat((init_embeds[node_name].to(args.device), 
					residual_node_embed), dim=1) for node_name, 
		residual_node_embed in checkpoint['residual_node_embeds'].items()}
		



		print('Finish loading checkpint')

		print('\n')


		print('Evaluating')


		all_score_dict = {}
		for task_idx, task_name in enumerate(args.task_name_list):


			task_type = args.task_type_list[task_idx]
			
			task_specific_output_layer = \
			task_specific_output_layer_list[task_idx]

			test_ids = test_id_list[task_idx]
			test_targets = test_target_list[task_idx]

			input_type_list = task_input_types[task_idx]

			if(task_type != 'ranking'):
				test_ids = torch.tensor(test_ids, 
					device=args.device).long()
				
				test_targets = torch.tensor(test_targets, 
					device=args.device).float()
				


			task_dir = args.task_dir_list[task_idx]


			if(task_type == 'classification'):#classification task

				test_score_dict = evaluate_classification_task(\
					test_ids, node_embeds, input_type_list,
					task_specific_output_layer, test_targets)


			elif(task_type == 'link_prediction'):#link prediction task

				test_score_dict = evaluate_link_prediction_task(
					test_ids, node_embeds, input_type_list, 
					task_specific_output_layer, test_targets,
					args.link_prediction_threshold, args.device)



			else:#ranking task   

				test_score_dict = evaluate_ranking_task(\
					test_ids, node_embeds, input_type_list, 
					task_specific_output_layer, test_targets, args.device)


			for metric, score in test_score_dict.items():
				key = task_name + ' ' + metric
				all_score_dict[key] = score

		all_score_df = pd.DataFrame.from_dict(all_score_dict, 
	orient='index', columns = ['Test'])

		
		

		all_score_df.to_csv(args.test_result_path)
		


		print('Finsih evaluating')
		print('\n')



	
	print('End of program')



if __name__ == '__main__':
	
	main()
