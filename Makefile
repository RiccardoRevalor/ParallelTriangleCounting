.PHONY: all\
		sequential_node_it.v1\
		sequential_node_it.v2\
		sequential_edge_it.v1\
		sequential_edge_it.v2\
		parallel_node_it_CPP.v1\
		parallel_node_it_CPP.v2\
		parallel_matrixmultiplication_CPP\
		parallel_edge_it_openmp\
		parallel_edge_it_manual_threads_CPP.v1\
		parallel_edge_it_manual_threads_CPP.v2\
		cuda_node_it.v1\
		cuda_node_it.v2\
		cuda_matrixmultiplication.v1\
		cuda_matrixmultiplication.v2\
		cuda_edge_it.v1\
		cuda_edge_it.v1_1\
		cuda_edge_it.v1_2\
		cuda_edge_it.v2\
		cuda_edge_it.v2_1\
		cuda_edge_it.v2_2\
		ORCHESTRATOR\
		clean

all:	sequential_node_it.v1\
	 	sequential_node_it.v2\
	 	sequential_edge_it.v1\
	 	sequential_edge_it.v2\
	 	parallel_node_it_CPP.v1\
	 	parallel_node_it_CPP.v2\
		parallel_matrixmultiplication_CPP\
		parallel_edge_it_openmp\
		parallel_edge_it_manual_threads_CPP.v1\
		parallel_edge_it_manual_threads_CPP.v2\
		cuda_node_it.v1\
		cuda_node_it.v2\
		cuda_matrixmultiplication.v1\
		cuda_matrixmultiplication.v2\
		cuda_edge_it.v1\
		cuda_edge_it.v1_1\
		cuda_edge_it.v1_2\
		cuda_edge_it.v2\
		cuda_edge_it.v2_1\
		cuda_edge_it.v2_2\
		ORCHESTRATOR\
		clean


# Compiler paths
nvcc_linux := /usr/local/cuda-12.9/bin/nvcc
nvcc_win := C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.9/bin/nvcc.exe

# cl.exe for windows
BUILD_CONFIG ?= M
VS_PATH_RICCARDO := "C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64"
VS_PATH_MICHELE := C:/Program\ Files/Microsoft\ Visual\ Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64

ifeq ($(BUILD_CONFIG),R)
    VS_PATH := $(VS_PATH_RICCARDO)
else ifeq ($(BUILD_CONFIG),M)
    VS_PATH := $(VS_PATH_MICHELE)
else
    $(error BUILD_CONFIG deve essere 'R' o 'M'. Esempio: make BUILD_CONFIG=R)
endif

OS ?= linux

ifeq ($(OS), linux)
	NVCC := $(nvcc_linux)
else ifeq ($(OS), windows)
	NVCC := $(nvcc_win)
else
    $(error Unsupported TARGET_OS value: '$(TARGET_OS)'. Use 'linux' or 'windows')
endif 

sequential_node_it.v1:
	$(MAKE) -C algorithms/sequential_node_it -f Makefile_v1

sequential_node_it.v2:
	$(MAKE) -C algorithms/sequential_node_it -f Makefile_v2

sequential_edge_it.v1:
	$(MAKE) -C algorithms/sequential_edge_it -f Makefile_v1

sequential_edge_it.v2:
	$(MAKE) -C algorithms/sequential_edge_it -f Makefile_v2

parallel_node_it_CPP.v1:
	$(MAKE) -C algorithms/parallel_node_it_CPP -f Makefile_v1

parallel_node_it_CPP.v2:
	$(MAKE) -C algorithms/parallel_node_it_CPP -f Makefile_v2

parallel_matrixmultiplication_CPP:
	$(MAKE) -C algorithms/parallel_matrixmultiplication_CPP -f Makefile_v1

parallel_edge_it_openmp:
	$(MAKE) -C algorithms/parallel_edge_it_openmp -f Makefile

parallel_edge_it_manual_threads_CPP.v1:
	$(MAKE) -C algorithms/parallel_edge_it_manual_threads_CPP -f Makefile_v1

parallel_edge_it_manual_threads_CPP.v2:
	$(MAKE) -C algorithms/parallel_edge_it_manual_threads_CPP -f Makefile_v2

cuda_node_it.v1:
	$(MAKE) -C algorithms/cuda_node_it -f Makefile_v1 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_node_it.v2:
	$(MAKE) -C algorithms/cuda_node_it -f Makefile_v2 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_matrixmultiplication.v1:
	$(MAKE) -C algorithms/cuda_matrixmultiplication -f Makefile_v1 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_matrixmultiplication.v2:
	$(MAKE) -C algorithms/cuda_matrixmultiplication -f Makefile_v2 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_edge_it.v1:
	$(MAKE) -C algorithms/cuda_edge_it -f Makefile_v1 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_edge_it.v1_1:
	$(MAKE) -C algorithms/cuda_edge_it -f Makefile_v1_1 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_edge_it.v1_2:
	$(MAKE) -C algorithms/cuda_edge_it -f Makefile_v1_2 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_edge_it.v2:
	$(MAKE) -C algorithms/cuda_edge_it -f Makefile_v2 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_edge_it.v2_1:
	$(MAKE) -C algorithms/cuda_edge_it -f Makefile_v2_1 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

cuda_edge_it.v2_2:
	$(MAKE) -C algorithms/cuda_edge_it -f Makefile_v2_2 OS=$(OS) NVCC=$(NVCC) VS_PATH=$(VS_PATH)

ifeq ($(OS), windows)
    MAKEFILE_NAME = Makefile_windows
else
    MAKEFILE_NAME = Makefile_linux
endif

ORCHESTRATOR:
	$(MAKE) -C CV_ORCHESTRATOR -f $(MAKEFILE_NAME)

clean:
ifeq ($(OS), windows)
	@cmd /C \"$(CURDIR)\delete_temps.bat\"t
else
	@echo Cleaning Linux/macOS-specific CUDA/linker artifacts...
	find algorithms CV_ORCHESTRATOR -name "*.exp" -delete
	find algorithms CV_ORCHESTRATOR -name "*.lib" -delete
	find algorithms CV_ORCHESTRATOR -name "*.pdb" -delete
	find algorithms CV_ORCHESTRATOR -name "*.ilk" -delete
endif
	@echo Clean process complete.