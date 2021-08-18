# Multi Relational Graph-based Heterogeneous Multi-Task Learning in Community Question Answering
This repository holds the code and data of our CIKM 2021 paper "Multi-Relational Graph based Heterogeneous Multi-TaskLearning in Community Question Answering".

# Library Requirements: 
1. Python 3.7
2. torch 1.4.0
3. dgl-cu90 0.4.3.post2

# Training:
1. Ensure that the following directories exist at the same level with src/: checkpoints/, log/, results/, data/, tensorboard_runs/;
2. Place the whole dataset into data/ directory;
3. In the src/ directory, run the following command:
```
CUDA_VISIBLE_DEVICES=$gpu_id \
DGLBACKEND=pytorch \
python -u main.py \
--max_num_epoch 135 \
--mode train \
--use_gpu \
--cet_list_path ../data/Stat/Clean_Data/Canonical_Edge_Types_List.json \
--edge_indices_dir ../data/Edge_Indices/ \
--node_name_list Clean_Tags Sample_Clean_Answers Sample_Clean_Questions Sample_Clean_Users \
--textual_feature_flag_list 1 1 1 1 \
--pre_trained_flag_list 0 1 1 1 \
--compressed_cf_embs_dir ../data/Compressed_Combined_Features_Embeddings/ \
--ntf_embs_dir ../data/Non_Textual_Features_Embeddings/ \
--pre_trained_emb_dir ../data/Pre_Trained_Embeddings/ \
--task_name_list Duplicate_Question_Detection Tag_Recommendation Answer_Recommendation Answer_Score_Classification User_Reputation_Classification \
--task_dir_list ../data/Duplicate_Question_Detection/ ../data/Tag_Recommendation/ ../data/Answer_Recommendation/ ../data/Answer_Score_Classification/ ../data/User_Reputation_Classification/ \
--task_type_list link_prediction link_prediction ranking classification classification \
--constraint_1_coefficient 1 \
--constraint_2_coefficient 1 \
--num_hidden_layer 0 \
--hidden_size_list 16 \
--constraint_1_list_path ../data/constraint_1_list.json \
--constraint_2_list_path ../data/constraint_2_list.json \
--checkpoint_path ../checkpoints/checkpoint.tar \
--num_mlp_layers_list 1 1 \
--batch_norm_flag_list 1 1 \
--activation_name_list leaky_relu leaky_relu \
--dropout_rate_list 0 0 \
--num_class_list 4 5 \
--train_dev_result_path ../results/Best_Train_Dev_Results.csv \
--test_result_path ../results/Test_Results.csv \
--link_prediction_pos_weight 2 \
--epsilon_list 0 0 \
--epsilon_trainable_list 0 0 \
--bias_flag_list 0 0 \
--l2_penalty_coef 1e-2 \
--lr_reduce_step 50 \
--tensorboard_log_dir ../tensorboard_runs/ \
--task_coef_list 7 1 1 7 1 \
> ../log/Train_stdout_Log.txt \
2> ../log/Train_stderr_Log.txt
```
4. After executing the program, the checkpoints with the best average scores on the development set will be saved in Checpoints/, the training log will be logged to log/, training results will be saved in results/, the tensorboard event logs will be saved in tensorboard_runs/;
5. See the comments within the soure code for more details about using the code.

# Dataset
1. The dataset is available via the link:   ;
2. See the Data_README.txt for the description of the dataset.
