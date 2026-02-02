# ------------------------------------------------------------
# Makefile — one executable per .cpp in src/
# ------------------------------------------------------------

CXX       := g++-14
CXXFLAGS  := -O2 -g -std=c++20 -march=native
INCLUDES  := -I/home/simo/eigen-3.4.0 \
             -I./fdaPDE-cpp \
             -I./fdaPDE-cpp/fdaPDE/core

SRC_DIR   := src
BUILD_DIR := build

# ------------------------------------------------------------------
# Discover all cpp sources and derive executable names
#   src/foo.cpp  ->  foo            (stem)
#   src/bar.cpp  ->  bar
# ------------------------------------------------------------------
SRC          := $(wildcard $(SRC_DIR)/*.cpp)
EXECUTABLES  := $(patsubst $(SRC_DIR)/%.cpp,%,$(SRC))          # foo bar …
TARGETS      := $(addprefix $(BUILD_DIR)/,$(EXECUTABLES))      # build/foo …

# ------------------------------------------------------------------
# Default target: build everything that was discovered
# ------------------------------------------------------------------
all: $(TARGETS)

# ------------------------------------------------------------------
# Pattern rule: build/foo  ←  src/foo.cpp
# $@ = build/foo   $< = src/foo.cpp
# ------------------------------------------------------------------
$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Ensure the build/ directory exists
$(BUILD_DIR):
	@mkdir -p $@

# ------------------------------------------------------------------
# Run every executable that was built
# ------------------------------------------------------------------
run: all
	@cd $(BUILD_DIR) && \
	for exe in $(EXECUTABLES); do \
	    echo ">>> running $$exe"; \
	    ./$$exe; \
	    echo; \
	done

valgrind: all
	@cd $(BUILD_DIR) && \
	for exe in $(EXECUTABLES); do \
	    echo ">>> running Valgrind on $$exe"; \
	    valgrind --leak-check=full --show-leak-kinds=all \
	             --track-origins=yes \
	             ./$$exe \
	             2> valgrind_$${exe}.txt; \
	    echo "log → valgrind_$${exe}.txt"; \
	    echo; \
	done

heaptrack: all
	@cd $(BUILD_DIR) && \
	for exe in $(EXECUTABLES); do \
	    echo ">>> running heaptrack on $$exe"; \
	    heaptrack ./$$exe \
	        2> heaptrack_$${exe}.txt; \
	    echo "log → heaptrack_$${exe}.txt"; \
	    echo; \
	done

# ------------------------------------------------------------------
# Clean up
# ------------------------------------------------------------------
clean:
	@rm -rf $(BUILD_DIR)

.PHONY: all run clean