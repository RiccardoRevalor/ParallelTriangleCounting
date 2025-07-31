import os
import pandas as pd
import matplotlib.pyplot as plt

#CSV input directories
CV_INPUT_DIR = "../cross_validation_output"
SEQUENTIAL_NODE_V1_INPUT_DIR = CV_INPUT_DIR + "/seq_node_it_v1/"
SEQUENTIAL_NODE_V2_INPUT_DIR = CV_INPUT_DIR + "/seq_node_it_v2/"
SEQUENTIAL_EDGE_V1_INPUT_DIR = CV_INPUT_DIR + "/seq_edge_it_v1/"
SEQUENTIAL_EDGE_V2_INPUT_DIR = CV_INPUT_DIR + "/seq_edge_it_v2/"
PARALLEL_NODE_V1_INPUT_DIR = CV_INPUT_DIR + "/parallel_node_it_v1/"
PARALLEL_NODE_V2_INPUT_DIR = CV_INPUT_DIR + "/parallel_node_it_v2/"
PARALLEL_EDGE_V1_INPUT_DIR = CV_INPUT_DIR + "/parallel_edge_it_manual_threads_v1/"
PARALLEL_EDGE_V2_INPUT_DIR = CV_INPUT_DIR + "/parallel_edge_it_manual_threads_v2/"
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



X_axis = ['NUM_THREADS']
Y_axis = ['TOTAL_DURATION_US']
desired_cols = X_axis + Y_axis

def readCSV(fileName, desired_cols):
    try:
        df = pd.read_csv(fileName, usecols=desired_cols)
        return df
    except Exception as e:
        print(f"Error reading {fileName}: {e}")
        return pd.DataFrame(columns=desired_cols)
    
def generateCharts(df, output_dir, X_axis, Y_axis, title=None):
    """
    Generate a 2D chart
    """
    if df.empty:
        print("DataFrame is empty. No charts to generate.")
        return
    
    for x_col in X_axis:
        for y_col in Y_axis:
            plt.figure(figsize=(10, 6))
            plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', color='red')
            plt.title(f'{y_col} vs {x_col}' if not title else title)

            if x_col == 'NUM_THREADS': x_label = 'Number of Threads'
            else: x_label = x_col

            if y_col == 'TOTAL_DURATION_US': y_label = 'Total Duration (µs)'
            else: y_label = y_col
              
            plt.xticks(df[x_col], rotation=45)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True)
            output_file = os.path.join(output_dir, f'{y_col}_vs_{x_col}.png' if not title else f'{title}.png')
            plt.savefig(output_file)
            plt.close()
            print(f"Chart saved to {output_file}")



def generateCharts_Line(df, output_dir, X_axis, title=None):
    """
    Generate a trend line (1D). Each column of the df will result in a separate line chart.
    """
    if df.empty:
        print("DataFrame is empty. No trend lines to generate.")
        return

    for col_name in X_axis: 
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping trend line for this column.")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[col_name], marker='o', linestyle='-', color='blue')


        chart_title = f'Trend of {col_name}' if not title else title
        plt.title(chart_title)


        plt.xlabel('Data Point Index') 


        y_label = col_name # Default label
        if col_name == 'NUM_THREADS':
            y_label = 'Number of Threads'
        elif col_name == 'TOTAL_DURATION_US':
            y_label = 'Total Duration (µs)'
        plt.ylabel(y_label)

        plt.grid(True)

        if not title:
            output_file_name = f'trend_{col_name}.png'
        else:
            output_file_name = f'{title.replace(" ", "_")}_trend_{col_name}.png'

        output_file = os.path.join(output_dir, output_file_name)
        plt.savefig(output_file)
        plt.close() 
        print(f"Trend line chart saved to {output_file}")




if __name__ == "__main__":
    input_file = PARALLEL_NODE_V2_INPUT_DIR + "graph_2ml_RTX_4060_M.csv" 
    output_dir = 'charts' 
    title = "Parallel Node (Forward) V2 - Graph 2 ML Nodes - Intel Core i7 14th gen"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = readCSV(input_file, desired_cols)
    generateCharts(df, output_dir, X_axis, Y_axis, title=title)