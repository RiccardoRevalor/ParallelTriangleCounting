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
		cuda_node_it.v2

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
		cuda_node_it.v2


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
	$(MAKE) -C algorithms/cuda_node_it -f Makefile_v1

cuda_node_it.v2:
	$(MAKE) -C algorithms/cuda_node_it -f Makefile_v2


