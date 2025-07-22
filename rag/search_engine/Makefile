# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -O3 -std=c++17

all: indexer HTTPServer #searchEngine 

indexer: indexer.cpp
	$(CXX) $(CXXFLAGS) -o indexer indexer.cpp

# searchEngine: searchEngine.cpp
# 	$(CXX) $(CXXFLAGS) -o searchEngine searchEngine.cpp

HTTPServer: HTTPServer.cpp searchEngine.cpp
	$(CXX) $(CXXFLAGS) -o HTTPServer HTTPServer.cpp searchEngine.cpp