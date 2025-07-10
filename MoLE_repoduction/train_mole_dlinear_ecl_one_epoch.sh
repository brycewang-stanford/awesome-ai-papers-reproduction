#!/bin/bash

# Script to train MoLE DLinear model on ECL dataset for one epoch and save the model

# Set default parameters
SEQ_LEN=336
PRED_LEN=96
LEARNING_RATE=0.0001
T_DIM=4
HEAD_DROPOUT=0.0
SEED=2021
BATCH_SIZE=8

# Create output directory
mkdir -p ./checkpoints/one_epoch_model/
mkdir -p ./logs/one_epoch_training/

# Set log file
LOG_FILE="./logs/one_epoch_training/ECL_MoLE_DLinear_one_epoch.log"

echo "Starting MoLE DLinear training for one epoch on ECL dataset..." | tee $LOG_FILE
echo "Parameters:" | tee -a $LOG_FILE
echo "SEQ_LEN: $SEQ_LEN" | tee -a $LOG_FILE
echo "PRED_LEN: $PRED_LEN" | tee -a $LOG_FILE
echo "LEARNING_RATE: $LEARNING_RATE" | tee -a $LOG_FILE
echo "T_DIM: $T_DIM" | tee -a $LOG_FILE
echo "HEAD_DROPOUT: $HEAD_DROPOUT" | tee -a $LOG_FILE
echo "BATCH_SIZE: $BATCH_SIZE" | tee -a $LOG_FILE
echo "SEED: $SEED" | tee -a $LOG_FILE
echo "$(date)" | tee -a $LOG_FILE

# Run training for one epoch
python -u run_longExp.py \
--is_training 1 \
--root_path ./dataset/ \
--data_path ECL.csv \
--model_id ECL_MoLE_DLinear_one_epoch \
--model MoLE_DLinear \
--data custom \
--features M \
--seq_len $SEQ_LEN \
--pred_len $PRED_LEN \
--enc_in 321 \
--des 'One_Epoch_Training' \
--itr 1 \
--batch_size $BATCH_SIZE \
--in_batch_augmentation \
--aug_method f_mask \
--aug_rate 0 \
--t_dim $T_DIM \
--head_dropout $HEAD_DROPOUT \
--seed $SEED \
--learning_rate $LEARNING_RATE \
--train_epochs 1 \
--num_workers 0 \
--checkpoints ./checkpoints/one_epoch_model/ 2>&1 | tee -a $LOG_FILE

echo "Training completed. Model saved in ./checkpoints/one_epoch_model/" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE