# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -O3 -std=c++17

all: search_HTTPServer_AWS

search_HTTPServer_AWS: search_HTTPServer_AWS.cpp searchEngine_AWS.cpp
	$(CXX) $(CXXFLAGS) -o search_HTTPServer_AWS search_HTTPServer_AWS.cpp searchEngine_AWS.cpp