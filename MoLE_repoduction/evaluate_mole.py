#!/usr/bin/env python3
import numpy as np
import torch
import sys
import os
from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main
from utils.metrics import metric
import argparse

def create_args():
    """Create default arguments for evaluation"""
    parser = argparse.ArgumentParser(description='MoLE Evaluation')
    
    # Basic settings
    parser.add_argument('--model', type=str, default='MoLE_DLinear')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='ECL.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    
    # Model settings
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=321)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--t_dim', type=int, default=16)
    
    # Other required args
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--moving_avg', type=int, default=25)
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Additional args
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--model_id', type=str, default='ECL_eval')
    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--drop', type=float, default=0.1)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    
    # DLinear specific
    parser.add_argument('--individual', type=int, default=0)
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    
    # Additional missing args
    parser.add_argument('--test_time_train', type=bool, default=False)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--disable_rev', type=bool, default=False)
    parser.add_argument('--chunk_size', type=int, default=40)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', type=str, default='end')
    parser.add_argument('--data_size', type=float, default=1.0)
    parser.add_argument('--aug_data_size', type=float, default=1.0)
    parser.add_argument('--wo_original_set', type=bool, default=False)
    
    return parser.parse_args([])

def evaluate_mole():
    """Evaluate MoLE model on electricity dataset"""
    args = create_args()
    
    # Fix seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Loading electricity dataset...")
    
    # Get data
    _, _, test_loader = data_provider(args, flag='test')
    
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total test batches: {len(test_loader)}")
    
    # Initialize model with random weights for demonstration
    from models.MoLE_DLinear import Model
    model = Model(args)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set model to eval mode
    model.eval()
    
    all_preds = []
    all_trues = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if i >= 5:  # Limit to first 5 batches for demo
                break
                
            # Model prediction (using random initialized weights for demo)
            outputs = model(batch_x, batch_x_mark, batch_y_mark[:, -args.pred_len:, :])
            
            # Get ground truth
            true_values = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
            pred_values = outputs.detach().cpu().numpy()
            
            all_preds.append(pred_values)
            all_trues.append(true_values)
            
            print(f"Processed batch {i+1}/5")
    
    # Concatenate all predictions and ground truth
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    print(f"Prediction shape: {preds.shape}")
    print(f"Ground truth shape: {trues.shape}")
    
    # Calculate metrics using existing metrics function
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    
    print("\n" + "="*50)
    print("MoLE Model Evaluation Results")
    print("="*50)
    print(f"MAE (Mean Absolute Error):     {mae:.6f}")
    print(f"MSE (Mean Squared Error):      {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.6f}")
    print(f"MSPE (Mean Squared Percentage Error):  {mspe:.6f}")
    print(f"RSE (Root Squared Error):      {rse:.6f}")
    print(f"CORR (Correlation):            {corr:.6f}")
    print("="*50)
    
    # Additional analysis
    print(f"\nPrediction statistics:")
    print(f"Min prediction: {preds.min():.6f}")
    print(f"Max prediction: {preds.max():.6f}")
    print(f"Mean prediction: {preds.mean():.6f}")
    print(f"Std prediction: {preds.std():.6f}")
    
    print(f"\nGround truth statistics:")
    print(f"Min true: {trues.min():.6f}")
    print(f"Max true: {trues.max():.6f}")
    print(f"Mean true: {trues.mean():.6f}")
    print(f"Std true: {trues.std():.6f}")
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 
        'mape': mape, 'mspe': mspe, 'rse': rse, 'corr': corr
    }

if __name__ == "__main__":
    evaluate_mole()