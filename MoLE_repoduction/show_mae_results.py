#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

def load_and_display_mae():
    """Load MAE results and display as DataFrame"""
    
    # Find the results folder
    results_base = './results/'
    if not os.path.exists(results_base):
        print("Results folder not found!")
        return
    
    # Get the experiment folder (should be only one)
    experiment_folders = [f for f in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, f))]
    
    if not experiment_folders:
        print("No experiment folders found!")
        return
    
    experiment_folder = experiment_folders[0]
    results_path = os.path.join(results_base, experiment_folder)
    
    print(f"Loading results from: {results_path}")
    print(f"Experiment: {experiment_folder}")
    print("=" * 80)
    
    # Load all metric files
    metrics_data = {}
    metric_files = ['mae_test.npy', 'mse_test.npy', 'rmse_test.npy', 'mape_test.npy', 'mspe_test.npy', 'rse_test.npy']
    
    for metric_file in metric_files:
        file_path = os.path.join(results_path, metric_file)
        if os.path.exists(file_path):
            try:
                value = np.load(file_path)
                metric_name = metric_file.replace('_test.npy', '').upper()
                metrics_data[metric_name] = float(value) if np.isscalar(value) else value
                print(f"✓ Loaded {metric_name}: {value}")
            except Exception as e:
                print(f"✗ Error loading {metric_file}: {e}")
        else:
            print(f"✗ File not found: {metric_file}")
    
    # Load correlation data (it's an array)
    corr_file = os.path.join(results_path, 'corr_test.npy')
    if os.path.exists(corr_file):
        try:
            corr_data = np.load(corr_file)
            print(f"✓ Loaded CORR: shape {corr_data.shape}")
            metrics_data['CORR_MEAN'] = float(np.mean(corr_data))
            metrics_data['CORR_STD'] = float(np.std(corr_data))
        except Exception as e:
            print(f"✗ Error loading correlation: {e}")
    
    print("\n" + "=" * 80)
    print("MOLE MODEL EVALUATION METRICS")
    print("=" * 80)
    
    # Create DataFrame from metrics
    if metrics_data:
        # Create a simple metrics summary DataFrame
        metrics_df = pd.DataFrame([metrics_data])
        
        print("📊 METRICS SUMMARY:")
        print(metrics_df.round(6).to_string(index=False))
        
        # Create a detailed DataFrame with metric descriptions
        metric_descriptions = {
            'MAE': 'Mean Absolute Error',
            'MSE': 'Mean Squared Error', 
            'RMSE': 'Root Mean Squared Error',
            'MAPE': 'Mean Absolute Percentage Error',
            'MSPE': 'Mean Squared Percentage Error',
            'RSE': 'Root Squared Error',
            'CORR_MEAN': 'Average Correlation',
            'CORR_STD': 'Correlation Standard Deviation'
        }
        
        detailed_data = []
        for metric, value in metrics_data.items():
            if metric in metric_descriptions:
                detailed_data.append({
                    'Metric': metric,
                    'Description': metric_descriptions[metric],
                    'Value': round(float(value), 6)
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        print(f"\n📋 DETAILED METRICS:")
        print(detailed_df.to_string(index=False))
        
        # Save DataFrame to CSV for easy viewing
        csv_path = os.path.join(results_path, 'metrics_summary.csv')
        detailed_df.to_csv(csv_path, index=False)
        print(f"\n💾 Metrics saved to: {csv_path}")
        
        return detailed_df
    else:
        print("No metrics data loaded!")
        return None

def load_prediction_summary():
    """Load and summarize prediction data"""
    
    results_base = './results/'
    experiment_folders = [f for f in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, f))]
    
    if not experiment_folders:
        return
    
    experiment_folder = experiment_folders[0]
    results_path = os.path.join(results_base, experiment_folder)
    
    print(f"\n" + "=" * 80)
    print("PREDICTION DATA SUMMARY")
    print("=" * 80)
    
    # Load prediction and true data for summary
    pred_file = os.path.join(results_path, 'pred_test.npy')
    true_file = os.path.join(results_path, 'true_test.npy')
    
    if os.path.exists(pred_file) and os.path.exists(true_file):
        try:
            preds = np.load(pred_file)
            trues = np.load(true_file)
            
            print(f"📊 Prediction Data Shape: {preds.shape}")
            print(f"📊 Ground Truth Shape: {trues.shape}")
            
            # Create summary statistics DataFrame
            summary_data = {
                'Dataset': ['Predictions', 'Ground Truth'],
                'Shape': [str(preds.shape), str(trues.shape)],
                'Min': [float(preds.min()), float(trues.min())],
                'Max': [float(preds.max()), float(trues.max())],
                'Mean': [float(preds.mean()), float(trues.mean())],
                'Std': [float(preds.std()), float(trues.std())]
            }
            
            summary_df = pd.DataFrame(summary_data)
            print(f"\n📈 DATA STATISTICS:")
            print(summary_df.round(6).to_string(index=False))
            
            # Save summary
            summary_csv_path = os.path.join(results_path, 'data_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\n💾 Data summary saved to: {summary_csv_path}")
            
            return summary_df
            
        except Exception as e:
            print(f"✗ Error loading prediction data: {e}")
    else:
        print("✗ Prediction or true data files not found!")
    
    return None

if __name__ == "__main__":
    print("🔍 MOLE Model Results Analysis")
    print("=" * 80)
    
    # Load and display metrics
    metrics_df = load_and_display_mae()
    
    # Load and display prediction summary
    summary_df = load_prediction_summary()
    
    print(f"\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)