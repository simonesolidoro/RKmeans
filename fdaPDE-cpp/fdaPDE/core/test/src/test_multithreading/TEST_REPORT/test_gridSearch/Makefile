CXX = g++-14 # Compilatore C++
CXXFLAGS = -std=c++20 -pthread -O3 -I/home/simo/eigen-3.4.0 -I../../../../../ # Flag di compilazione con include 
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
