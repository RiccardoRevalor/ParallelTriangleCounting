@echo off
set "original_dir=%cd%"

:: Set the target directory
set "target_dir=%~dp0cross_validation_output"

:: Check if cross_validation_output exists; if not, create it
if not exist "%target_dir%" (
    mkdir "%target_dir%"
)

:: Move into the directory
cd /d "%target_dir%"

:: Create all subfolders
mkdir seq_node_it_v1
mkdir seq_node_it_v2
mkdir seq_edge_it_v1
mkdir seq_edge_it_v2
mkdir parallel_node_it_v1
mkdir parallel_node_it_v2
mkdir parallel_matrixmultiplication
mkdir parallel_edge_it_manual_threads_v1
mkdir parallel_edge_it_manual_threads_v2
mkdir cuda_node_it_v1
mkdir cuda_node_it_v2
mkdir cuda_matrixmultiplication_v1
mkdir cuda_matrixmultiplication_v2
mkdir cuda_edge_it_v1
mkdir cuda_edge_it_v1_1
mkdir cuda_edge_it_v1_2
mkdir cuda_edge_it_v2
mkdir cuda_edge_it_v2_1
mkdir cuda_edge_it_v2_2

:: Return to the original directory
cd /d "%original_dir%"

echo All folders created successfully.
pause
