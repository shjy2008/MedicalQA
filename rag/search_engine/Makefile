# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -O3 -std=c++17

all: indexer search_HTTPServer #searchEngine 

indexer: indexer.cpp
	$(CXX) $(CXXFLAGS) -o indexer indexer.cpp

# searchEngine: searchEngine.cpp
# 	$(CXX) $(CXXFLAGS) -o searchEngine searchEngine.cpp

search_HTTPServer: search_HTTPServer.cpp searchEngine.cpp
	$(CXX) $(CXXFLAGS) -o search_HTTPServer search_HTTPServer.cpp searchEngine.cpp