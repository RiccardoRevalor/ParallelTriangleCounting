import os
import pandas as pd
import matplotlib.pyplot as plt

#CSV input directories
CV_INPUT_DIR = "../cross_validation_output"
CUDA_MATRIXMULTIPLICATION_V1_INPUT_DIR = CV_INPUT_DIR + "/cuda_matrixmultiplication_v1/"
CUDA_MATRIXMULTIPLICATION_V2_INPUT_DIR = CV_INPUT_DIR + "/cuda_matrixmultiplication_v2/"

# X and Y axis for default single-line plots
X_axis = ['BLOCK_SIZE']
Y_axis = ['TOTAL_DURATION_US']
desired_cols = ['TILE_SIZE', 'TRACE_BLOCKSIZE', 'TOTAL_DURATION_US']

def readCSV(fileName, desired_cols):
    try:
        df = pd.read_csv(fileName, usecols=desired_cols)
        return df
    except Exception as e:
        print(f"Error reading {fileName}: {e}")
        return pd.DataFrame(columns=desired_cols)


def generateCharts(df, output_dir, X_axis, Y_axis, title=None):
    """
    Generate 2D charts with single lines
    """
    if df.empty:
        print("DataFrame is empty. No charts to generate.")
        return
    
    for x_col in X_axis:
        for y_col in Y_axis:
            plt.figure(figsize=(10, 6))
            plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', color='red')
            plt.title(f'{y_col} vs {x_col}' if not title else title)

            x_label = 'Number of Threads' if x_col == 'NUM_THREADS' else x_col
            y_label = 'Total Duration (µs)' if y_col == 'TOTAL_DURATION_US' else y_col

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
    Generate trend line charts for individual columns
    """
    if df.empty:
        print("DataFrame is empty. No trend lines to generate.")
        return

    for col_name in X_axis: 
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping trend line.")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[col_name], marker='o', linestyle='-', color='blue')
        chart_title = f'Trend of {col_name}' if not title else title
        plt.title(chart_title)

        y_label = {
            'NUM_THREADS': 'Number of Threads',
            'BLOCK_SIZE': 'number of blocks',
            'TOTAL_DURATION_US': 'Total Duration (µs)'
        }.get(col_name, col_name)

        plt.xlabel('Data Point Index')
        plt.ylabel(y_label)
        plt.grid(True)

        output_file_name = f'trend_{col_name}.png' if not title else f'{title.replace(" ", "_")}_trend_{col_name}.png'
        output_file = os.path.join(output_dir, output_file_name)
        plt.savefig(output_file)
        plt.close()
        print(f"Trend line chart saved to {output_file}")


def generateMultiLineChart(df, output_dir, title=None):
    """
    Generate a multi-line chart of TOTAL_DURATION_US vs TILE_SIZE
    for each unique TRACE_BLOCKSIZE.
    """
    if df.empty:
        print("DataFrame is empty. No multi-line chart generated.")
        return

    plt.figure(figsize=(10, 6))

    for trace_blocksize in sorted(df['TRACE_BLOCKSIZE'].unique()):
        subset = df[df['TRACE_BLOCKSIZE'] == trace_blocksize]
        subset = subset.sort_values(by='TILE_SIZE')

        plt.plot(
            subset['TILE_SIZE'],
            subset['TOTAL_DURATION_US'],
            marker='o',
            linestyle='-',
            label=f'TRACE_BLOCKSIZE={trace_blocksize}'
        )

    plt.title(title or "Total Duration vs Tile Size by Trace Blocksize")
    plt.xlabel("Tile Size")
    plt.ylabel("Total Duration (µs)")
    plt.legend(title="Trace Block Size")
    plt.grid(True)
    plt.xticks(sorted(df['TILE_SIZE'].unique()))

    output_file = os.path.join(output_dir, 'multi_line_total_duration_vs_tile_size.png')
    plt.savefig(output_file)
    plt.close()
    print(f"Multi-line chart saved to {output_file}")


if __name__ == "__main__":
    input_file = CUDA_MATRIXMULTIPLICATION_V2_INPUT_DIR + "graph_10k_RTX_4060_M.csv"
    output_dir = 'charts'
    title = "Cuda Matrix Multiplicaiton V2 - Graph 10k Nodes - RTX 4060 Laptop GPU"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = readCSV(input_file, desired_cols)
    generateMultiLineChart(df, output_dir, title=title)