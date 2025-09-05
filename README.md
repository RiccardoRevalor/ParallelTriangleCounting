## About the Project
![Introduction Picture](./img/intro_pic.jpg)

This project evaluates the performance of different implementations of the Forward Algorithm for counting triangles in a graph. We provide both a sequential baseline and parallelized versions, where parallelization is achieved using CPU multithreading and GPU acceleration.

## Getting started

The following is a complete guide on how to set up and run the program correctly on a windows machine.

### Prerequisites
- Computer with a windows operating system
- NVIDIA GPU that can support the CUDA toolkit.
- At least 16GB of ram
- make and g++.
- cl.exe from Visual Studio Tools.

The setup phase of the CUDA toolkit, make and g++ is not covered in this guide. 

### set up variables in make file

- copy the path to nvcc.exe
- open the Makefile in the project folder
- paste the path between "" of the variable `nvcc_win`

![var_2](./img/var_2.jpg)

- Copy the path to cl.exe.
- open the Makefile in the project folder
- paste the path between "" of the variable `VS_PATH_USER`

![var_1](./img/var_1.jpg)

### Compilation

#### Compile algorithms
The Makefile in the project folder automatically compiles all the algorithms.
Run the following command to compile the algorithms:
```
make OS=windows BUILD_CONFIG=U
```

![main_makefile](./img/main_makefile.jpg)

#### Compile Orchestrator
The Orchestrator is the program located in the folder `CV_ORCHESTRATOR` that tests the algorithms all in one single run and create csv files for each algorithm about the performance in the folder `cross_validation_output`. In the subdirectory CV_ORCHESTRATOR run the following program to compile orchestrator_windows.cpp
```
make OS=windows
``` 

![orchestrator_makefile](./img/orchestrator_makefile.jpg)

### run orchestrator
In the folder `CV_ORCHESTRATOR` run the orchestrator through the following command, replacing <GPU_MODEL> with the name of your GPU:
```
.\orchestrator_windows.exe <GPU_MODEL>
```

### results
Each algorithm saves its performance in a specific folder (distinguished by its name) in the folder cross_validation_output through a csv file. The csv file name has the following format: `graphName_GPU_User.csv`.