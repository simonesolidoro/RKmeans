CXX = g++-14 # Compilatore C++
CXXFLAGS = -I./../../../fdaPDE-cpp -I./../../../fdaPDE-cpp/fdaPDE/core -I/home/simo/eigen-3.4.0 -O2 -march=native -std=c++20 -s 
SOURCES = $(wildcard test_*.cpp) 
EXECUTABLES = $(SOURCES:.cpp=) 

all: $(EXECUTABLES)


%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@


clean:
	rm -f $(EXECUTABLES)
