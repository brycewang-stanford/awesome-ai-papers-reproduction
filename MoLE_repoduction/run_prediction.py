#!/usr/bin/env python3
import os
import numpy as np
import torch
from exp.exp_main import Exp_Main
import argparse

def create_args():
    """Create arguments for prediction"""
    parser = argparse.ArgumentParser()
    
    # Basic settings
    parser.add_argument('--is_training', type=int, default=0)
    parser.add_argument('--model_id', type=str, default='ECL_336_96_prediction')
    parser.add_argument('--model', type=str, default='MoLE_DLinear')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='ECL.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Model parameters
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=321)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--t_dim', type=int, default=16)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--drop', type=float, default=0.1)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--embed_type', type=int, default=0)
    
    # Training settings
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--pct_start', type=float, default=0.3)
    
    # GPU settings
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    
    # Additional required args
    parser.add_argument('--individual', type=int, default=0)
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--disable_rev', type=bool, default=False)
    parser.add_argument('--chunk_size', type=int, default=40)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', type=str, default='end')
    
    # Additional flags
    parser.add_argument('--test_flop', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)
    parser.add_argument('--save_gating_weights', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--test_time_train', type=bool, default=False)
    parser.add_argument('--data_size', type=float, default=1.0)
    parser.add_argument('--aug_data_size', type=float, default=1.0)
    parser.add_argument('--wo_original_set', type=bool, default=False)
    parser.add_argument('--in_dataset_augmentation', type=bool, default=False)
    parser.add_argument('--in_batch_augmentation', type=bool, default=False)
    parser.add_argument('--aug_method', type=str, default='f_mask')
    parser.add_argument('--aug_rate', type=float, default=0.0)
    parser.add_argument('--closer_data_aug_more', type=bool, default=False)
    
    return parser.parse_args([])

def main():
    """Run prediction and save results"""
    args = create_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Setting up MoLE model for prediction...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data_path}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prediction length: {args.pred_len}")
    
    # Create experiment
    exp = Exp_Main(args)
    
    # Create setting string
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_ebtimeF_dtTrue_Exp_0'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.fc_dropout)
    
    print(f"Running prediction with setting: {setting}")
    
    # Run test (this will save results to ./results/ folder)
    exp.test(setting, test=0)
    
    # Check if results were saved
    results_folder = f'./results/{setting}/'
    if os.path.exists(results_folder):
        print(f"\nResults saved to: {results_folder}")
        files = os.listdir(results_folder)
        print("Generated files:")
        for file in files:
            file_path = os.path.join(results_folder, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
                
                # If it's a numpy file, show its shape
                if file.endswith('.npy'):
                    try:
                        data = np.load(file_path)
                        print(f"    Shape: {data.shape}, Type: {data.dtype}")
                    except:
                        print(f"    Could not load numpy file")
    else:
        print(f"Results folder not found: {results_folder}")
    
    # Also check for result.txt
    if os.path.exists('result.txt'):
        print(f"\nMetrics summary saved to: result.txt")
        with open('result.txt', 'r') as f:
            lines = f.readlines()
            if lines:
                print("Last few lines of result.txt:")
                for line in lines[-5:]:
                    print(f"  {line.strip()}")

if __name__ == "__main__":
    main()