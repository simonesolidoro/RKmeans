# Variabili
CXX = g++ # Compilatore C++
CXXFLAGS = -std=c++20 -L/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/tbb/2021.3.0/lib -ltbb -pthread -Wall -I../../../../../ # Flag di compilazione con include
SOURCES = $(wildcard test_*.cpp) # Trova tutti i file test_*.cpp
EXECUTABLES = $(SOURCES:.cpp=) # Rinomina i file .cpp in .exe

# Obiettivo di default
all: $(EXECUTABLES)

# Regola per compilare ciascun file test_*.cpp
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Pulizia dei file generati
clean:
	rm -f $(EXECUTABLES)
