ifeq ($(OS), windows)
run: orchestrator_windows.cpp
	g++ -std=c++20 -cpp orchestrator_windows.cpp -o orchestrator_windows.exe

else ifeq ($(OS), linux)
run: orchestrator_linux.cpp
	g++ -std=c++20 -cpp orchestrator_linux.cpp -o orchestrator_linux

endif