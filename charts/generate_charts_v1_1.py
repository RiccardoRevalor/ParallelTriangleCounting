import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV input directories (same as before)
CV_INPUT_DIR = "../cross_validation_output"
SEQUENTIAL_NODE_V1_INPUT_DIR = CV_INPUT_DIR + "/seq_node_it_v1/"
SEQUENTIAL_NODE_V2_INPUT_DIR = CV_INPUT_DIR + "/seq_node_it_v2/"
SEQUENTIAL_EDGE_V1_INPUT_DIR = CV_INPUT_DIR + "/seq_edge_it_v1/"
SEQUENTIAL_EDGE_V2_INPUT_DIR = CV_INPUT_DIR + "/seq_edge_it_v2/"
PARALLEL_NODE_V1_INPUT_DIR = CV_INPUT_DIR + "/parallel_node_it_v1/"
PARALLEL_NODE_V2_INPUT_DIR = CV_INPUT_DIR + "/parallel_node_it_v2/"
PARALLEL_NODE_V3_INPUT_DIR = CV_INPUT_DIR + "/parallel_node_it_v3/"
PARALLEL_EDGE_V1_INPUT_DIR = CV_INPUT_DIR + "/parallel_edge_it_manual_threads_v1/"
PARALLEL_EDGE_V2_INPUT_DIR = CV_INPUT_DIR + "/parallel_edge_it_manual_threads_v2/"
PARALLEL_EDGE_V3_INPUT_DIR = CV_INPUT_DIR + "/parallel_edge_it_manual_threads_v3/"
PARALLEL_MATRIXMULTIPLICATION_INPUT_DIR = CV_INPUT_DIR + "/parallel_matrixmultiplication/"
CUDA_NODE_V1_INPUT_DIR = CV_INPUT_DIR + "/cuda_node_it_v1/"
CUDA_NODE_V2_INPUT_DIR = CV_INPUT_DIR + "/cuda_node_it_v2/"
CUDA_EDGE_V1_INPUT_DIR = CV_INPUT_DIR + "/cuda_edge_it_v1/"
CUDA_EDGE_V1_1_INPUT_DIR = CV_INPUT_DIR + "/cuda_edge_it_v1_1/"
CUDA_EDGE_V1_2_INPUT_DIR = CV_INPUT_DIR + "/cuda_edge_it_v1_2/"
CUDA_EDGE_V2_INPUT_DIR = CV_INPUT_DIR + "/cuda_edge_it_v2/"
CUDA_EDGE_V2_1_INPUT_DIR = CV_INPUT_DIR + "/cuda_edge_it_v2_1/"
CUDA_EDGE_V2_2_INPUT_DIR = CV_INPUT_DIR + "/cuda_edge_it_v2_2/"
CUDA_MATRIXMULTIPLICATION_V1_INPUT_DIR = CV_INPUT_DIR + "/cuda_matrixmultiplication_v1/"
CUDA_MATRIXMULTIPLICATION_V2_INPUT_DIR = CV_INPUT_DIR + "/cuda_matrixmultiplication_v2/"

# Columns for chart axes
X_axis = ['MAX_SHARED_LIST_PER_EDGE_COMBINED']
Y_axis = ['TOTAL_DURATION_US']
GROUP_AXIS = 'BLOCK_SIZE'  # Each line in chart represents a unique BLOCK_SIZE

def readCSV(fileName):
    try:
        df = pd.read_csv(fileName)
        return df
    except Exception as e:
        print(f"Error reading {fileName}: {e}")
        return pd.DataFrame()

def generateMultiLineChart(df, output_dir, x_col, y_col, group_col, title=None):
    """
    Generate a multi-line chart. Each line represents a group in group_col.
    """
    if df.empty:
        print("DataFrame is empty. No chart to generate.")
        return

    plt.figure(figsize=(12, 7))
    
    for group_value, group_df in df.groupby(group_col):
        sorted_df = group_df.sort_values(by=x_col)
        plt.plot(sorted_df[x_col], sorted_df[y_col], marker='o', label=f'{group_col}: {group_value}')
    
    chart_title = f'{y_col} vs {x_col} (grouped by {group_col})' if not title else title
    plt.title(chart_title)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel('Total Duration (µs)' if y_col == 'TOTAL_DURATION_US' else y_col)
    plt.legend(title=group_col.replace('_', ' ').title())
    plt.grid(True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f'{y_col}_vs_{x_col}_grouped_by_{group_col}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Multi-line chart saved to {output_path}")

if __name__ == "__main__":
    input_file = CUDA_EDGE_V2_2_INPUT_DIR + "graph_100ml_RTX_4060_M.csv"
    output_dir = "charts"
    title = "CUDA Edge V2_2 - Total Duration vs Max Shared - Grouped by Block Size"

    df = readCSV(input_file)
    generateMultiLineChart(df, output_dir, x_col='MAX_SHARED_LIST_PER_EDGE_COMBINED', y_col='TOTAL_DURATION_US', group_col='BLOCK_SIZE', title=title)
