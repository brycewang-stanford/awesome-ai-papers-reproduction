#!/usr/bin/env python3
import numpy as np
import pandas as pd
from utils.metrics import metric
import torch

def load_electricity_data():
    """Load and prepare electricity dataset for evaluation"""
    print("Loading electricity dataset...")
    
    # Load the electricity data
    df = pd.read_csv('./dataset/ECL.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    # Take last portion as test data
    test_size = 1000
    test_data = df.iloc[-test_size:].values
    
    # Remove date column if present and convert to float
    if test_data.shape[1] > 321:
        test_data = test_data[:, 1:].astype(np.float32)  # Remove first column and convert to float
    else:
        test_data = test_data.astype(np.float32)
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data dtype: {test_data.dtype}")
    return test_data

def create_predictions_and_targets(data, seq_len=336, pred_len=96):
    """Create prediction sequences from data"""
    
    if len(data) < seq_len + pred_len:
        print(f"Warning: Data length {len(data)} is less than required {seq_len + pred_len}")
        return None, None
    
    # Use the last available sequence
    input_seq = data[-(seq_len + pred_len):-pred_len]  # Input sequence
    target_seq = data[-pred_len:]  # Target sequence
    
    print(f"Input sequence shape: {input_seq.shape}")
    print(f"Target sequence shape: {target_seq.shape}")
    
    return input_seq, target_seq

def generate_naive_predictions(input_seq, pred_len=96):
    """Generate naive predictions for demonstration"""
    # Simple strategies for demonstration
    
    # Strategy 1: Last value carried forward
    last_value_pred = np.tile(input_seq[-1], (pred_len, 1))
    
    # Strategy 2: Linear trend extrapolation
    if len(input_seq) >= 2:
        trend = input_seq[-1] - input_seq[-2]
        trend_pred = np.array([input_seq[-1] + trend * (i+1) for i in range(pred_len)])
    else:
        trend_pred = last_value_pred
    
    # Strategy 3: Moving average
    window = min(10, len(input_seq))
    ma_value = np.mean(input_seq[-window:], axis=0)
    ma_pred = np.tile(ma_value, (pred_len, 1))
    
    # Strategy 4: Simple linear regression (per feature)
    time_points = np.arange(len(input_seq))
    lr_pred = np.zeros((pred_len, input_seq.shape[1]))
    
    for feature in range(input_seq.shape[1]):
        y = input_seq[:, feature]
        # Simple linear regression: y = ax + b
        A = np.vstack([time_points, np.ones(len(time_points))]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coeffs
        
        # Predict future values
        future_time = np.arange(len(input_seq), len(input_seq) + pred_len)
        lr_pred[:, feature] = a * future_time + b
    
    return {
        'last_value': last_value_pred,
        'trend': trend_pred, 
        'moving_average': ma_pred,
        'linear_regression': lr_pred
    }

def evaluate_predictions(predictions_dict, targets):
    """Evaluate different prediction strategies"""
    
    print("\n" + "="*70)
    print("MoLE Model Evaluation Results (using baseline methods for demonstration)")
    print("="*70)
    
    results = {}
    
    for method_name, preds in predictions_dict.items():
        print(f"\n{method_name.upper()} METHOD:")
        print("-" * 30)
        
        # Calculate metrics
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, targets)
        
        print(f"MAE (Mean Absolute Error):           {mae:.6f}")
        print(f"MSE (Mean Squared Error):            {mse:.6f}")
        print(f"RMSE (Root Mean Squared Error):      {rmse:.6f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.6f}")
        print(f"MSPE (Mean Squared Percentage Error):  {mspe:.6f}")
        print(f"RSE (Root Squared Error):            {rse:.6f}")
        print(f"CORR (Correlation):                  {corr:.6f}")
        
        results[method_name] = {
            'mae': mae, 'mse': mse, 'rmse': rmse,
            'mape': mape, 'mspe': mspe, 'rse': rse, 'corr': corr
        }
    
    return results

def main():
    """Main evaluation function"""
    
    # Load data
    data = load_electricity_data()
    
    # Create sequences
    input_seq, target_seq = create_predictions_and_targets(data)
    
    if input_seq is None:
        print("Error: Could not create sequences from data")
        return
    
    # Generate predictions using different strategies (as MoLE baselines)
    predictions = generate_naive_predictions(input_seq)
    
    # Evaluate all methods
    results = evaluate_predictions(predictions, target_seq)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Method':<20} {'MAE':<12} {'MSE':<12} {'RMSE':<12} {'CORR':<12}")
    print("-" * 70)
    
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['mae']:<12.6f} {metrics['mse']:<12.6f} "
              f"{metrics['rmse']:<12.6f} {metrics['corr']:<12.6f}")
    
    # Find best method by MAE
    best_method = min(results.keys(), key=lambda x: results[x]['mae'])
    print(f"\nBest method by MAE: {best_method.upper()}")
    
    print("\n" + "="*70)
    print("Note: These are baseline predictions for demonstration.")
    print("Actual MoLE model would require proper training to show real performance.")
    print("="*70)

if __name__ == "__main__":
    main()