# # Compiler
# CXX = g++

# # Compiler flags
# CXXFLAGS = -Wall -Wextra -O3 -std=c++17

# all: search_HTTPServer_AWS

# search_HTTPServer_AWS: search_HTTPServer_AWS.cpp searchEngine_AWS.cpp
# 	$(CXX) $(CXXFLAGS) -o search_HTTPServer_AWS search_HTTPServer_AWS.cpp searchEngine_AWS.cpp


# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 -pthread

# AWS SDK paths
AWS_INC = /usr/local/include    # AWS SDK include path
AWS_LIB = /usr/local/lib64      # AWS SDK library path
AWS_LIBS = -laws-cpp-sdk-core -laws-cpp-sdk-s3 # AWS SDK libraries

# Target
TARGET = search_HTTPServer_AWS

# Source files
SRCS = search_HTTPServer_AWS.cpp searchEngine_AWS.cpp ../searchEngine.cpp

# Default rule
all: $(TARGET)

# Build target with rpath to AWS SDK libraries
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -I$(AWS_INC) $(SRCS) \
	-o $(TARGET) -L$(AWS_LIB) -Wl,-rpath,$(AWS_LIB) $(AWS_LIBS)

# Clean
clean:
	rm -f $(TARGET)

