# Nome del compilatore C++
CXX = g++

# Number of threads to use for OpenMP
OMP_NUM_THREADS = 2

# Flag per il compilatore:
# -std=c++17: Usa lo standard C++17
# -O3: Massima ottimizzazione per la velocità
# -fopenmp: Abilita il supporto a OpenMP (FONDAMENTALE)
# -Wall: Abilita tutti i warning principali (buona pratica)
CXXFLAGS = -std=c++20 -O3 -fopenmp -Wall

# Nome del file sorgente (cambialo se il tuo file ha un nome diverso)
SRCS = $(wildcard *.cpp) ../utils/utils.cpp ../utils/matrixMath.cpp

# Nome del file eseguibile che verrà creato
TARGET = main

# --- REGOLE ---

# La regola di default: cosa fa 'make' quando eseguito senza argomenti
all: $(TARGET)

# Regola per creare l'eseguibile a partire dai sorgenti
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Regola per eseguire il programma
# Prima compila se necessario, poi imposta le variabili d'ambiente e lo lancia
run: $(TARGET)
	@echo "--- Esecuzione con $(OMP_NUM_THREADS) thread ---"
	export OMP_NUM_THREADS=$(OMP_NUM_THREADS) && ./$(TARGET)

# Regola per pulire la cartella dai file compilati
clean:
	rm -f $(TARGET)

# Specifica che 'all', 'run', e 'clean' non sono nomi di file
.PHONY: all run clean