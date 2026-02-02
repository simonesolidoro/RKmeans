CXX = g++-14 # Compilatore C++
CXXFLAGS = -std=c++20 -pthread -O3 -I/home/simo/eigen-3.4.0 -I../../../../../ 
SOURCES = $(wildcard test_*.cpp) 
EXECUTABLES = $(SOURCES:.cpp=) 

all: $(EXECUTABLES)


%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@


clean:
	rm -f $(EXECUTABLES)

