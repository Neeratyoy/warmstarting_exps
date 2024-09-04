import re
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from saws.config.yaml_utils import path_constructor
from tbparse import SummaryReader

BEST_SMOOTHING_WINDOW = 1024 * 3

def parse_yaml(file_path):
    with file_path.open('r') as file:
        content = yaml.safe_load(file)
    return content

def extract_params_from_dirname(dirname):
    pattern = r'lr=([\d\.]+)_batch_size=(\d+)'
    match = re.search(pattern, str(dirname))
    if match:
        lr = float(match.group(1))
        batch_size = int(match.group(2))
        return lr, batch_size
    return None, None

def collect_data(search_dir, loss_type='val'):
    if loss_type == 'val':
        result_key = 'val_loss'
        tb_key = 'Validation Loss'
    elif loss_type == 'train':
        result_key = 'train_loss'
        tb_key = 'Train Loss'
    else:
        raise ValueError("loss_type must be either 'val' or 'train'")
    
    data_result = []
    data_best_smooth_loss = []
    data_last_smooth_loss = []
    search_path = Path(search_dir)
    for dir_path in search_path.iterdir():
        if dir_path.is_dir():
            lr, batch_size = extract_params_from_dirname(dir_path.name)
            if lr is not None and batch_size is not None:
                yaml_path = dir_path / 'result.yaml'
                parquet_file = dir_path / "tb_logs.parquet"
                training_finished = (dir_path / 'tb_logs.csv').exists()
                if training_finished:
                    result = parse_yaml(yaml_path)
                    data_result.append((lr, batch_size, result[result_key]))
                    tb_logs = pd.read_parquet(parquet_file, columns=[tb_key, 'step', 'warmstart_stage'])
                    tb_logs = tb_logs[tb_logs['warmstart_stage'] == tb_logs['warmstart_stage'].max()]
                    tb_logs.dropna(inplace=True, axis=0)
                    tb_logs.sort_values(by='step', inplace=True)
                    values = tb_logs[tb_key].rolling(int(BEST_SMOOTHING_WINDOW / batch_size)).mean()
                    data_best_smooth_loss.append((lr, batch_size, values.min()))
                    data_last_smooth_loss.append((lr, batch_size, values.iloc[-1]))
                    
    # Convert data to DataFrame for easier manipulation
    df_result = pd.DataFrame(data_result, columns=['lr', 'batch_size', 'loss'])
    df_best_smooth = pd.DataFrame(data_best_smooth_loss, columns=['lr', 'batch_size', 'loss'])
    df_last_smooth = pd.DataFrame(data_last_smooth_loss, columns=['lr', 'batch_size', 'loss'])
    
    # Pivot the DataFrame to get the format needed for a heatmap
    heatmap_data_result = df_result.pivot(index='batch_size', columns='lr', values='loss')
    heatmap_data_best_smooth = df_best_smooth.pivot(index='batch_size', columns='lr', values='loss')
    heatmap_data_last_smooth = df_last_smooth.pivot(index='batch_size', columns='lr', values='loss')
    return heatmap_data_result, heatmap_data_best_smooth, heatmap_data_last_smooth

def create_heatmap(heatmap_data, output_path, file_name):
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="viridis", ax=ax)
    ax.set_title('Validation Loss Heatmap')
    ax.set_xlabel('Learning Rate (lr)')
    ax.set_ylabel('Batch Size')
    
    # Save the figure
    output_file = output_path / 'figures' / file_name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)

    min_value = heatmap_data.min().min()
    min_batch, min_lr = heatmap_data.stack().idxmin()
    
    print(f"-------- Results for {file_name} --------")
    print("The lowest loss value is:", min_value)
    print("Learning rate of the lowest value:", min_lr)
    print("Batch size of the lowest value:", min_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for result.yaml files and create a heatmap of validation losses.")
    parser.add_argument('--search_dir', type=str, help='The directory to search for result.yaml files')
    parser.add_argument('--lrs', 
                        type=float, 
                        nargs='+', 
                        default=[0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001], 
                        help='lrs used in construction. Need to be in the same order as the grid search')
    parser.add_argument('--batch_sizes', 
                        type=float, 
                        nargs='+', 
                        default=[16, 32, 64, 128, 256, 512, 1024], 
                        help='batch sizes used in construction. Need to be in the same order as the grid search')
    parser.add_argument('--loss_type', type=str, help='The kind of loss, either "val" or "train"', default='val', choices=['val', 'train'])

    
    args = parser.parse_args()
    search_dir = args.search_dir
    
    heatmap_data_result, heatmap_data_best_smooth, heatmap_data_last_smooth = collect_data(search_dir, args.loss_type)

    data = heatmap_data_result
    missing_indices = []
    for j, batch_size in enumerate(args.batch_sizes):
        for i, lr in enumerate(args.lrs):
            if lr not in data.columns or batch_size not in data.index or pd.isna(data.at[batch_size, lr]):
                missing_indices.append(str(i+j*len(args.lrs)))
    
    if len(missing_indices) > 0:
        print(f"A total of {len(missing_indices)} experiments did not run:")
        print(f"Missing indices: {','.join(missing_indices)}")
    else:
        print("All experiments ran.")
    
    if len(data)>0:
        create_heatmap(heatmap_data_result, Path(search_dir), f"{args.loss_type}_loss_heatmap_result.png")
        create_heatmap(heatmap_data_best_smooth, Path(search_dir), f"{args.loss_type}_loss_heatmap_best_smooth.png")
        create_heatmap(heatmap_data_last_smooth, Path(search_dir), f"{args.loss_type}_loss_heatmap_last_smooth.png")
    else:
        print("No valid result.yaml files found.")
