CXX = g++-13
CXXFLAGS = -std=c++17 -g -O3 -march=native -funroll-loops -ftree-vectorize -fopenmp
LDFLAGS = -Wl,-ld_classic -fopenmp

# Source files for each target
TENSOR_SOURCES = \
    ./tensor/Tensor.cpp \
    ./utill/Print.cpp \
    ./tests/TensorTests.cpp

LEARNING_SOURCES = \
    ./tensor/Tensor.cpp \
    ./utill/Print.cpp \
    ./loss/Loss.cpp \
    ./strategy/Strategy.cpp \
    ./model/Model.cpp \
    ./examples/Learning.cpp

MNIST_SOURCES = \
    ./tensor/Tensor.cpp \
    ./utill/Print.cpp \
    ./loss/Loss.cpp \
    ./strategy/Strategy.cpp \
    ./model/Model.cpp \
    ./examples/MNIST.cpp

# Object files for each target
TENSOR_OBJS = $(TENSOR_SOURCES:.cpp=.o)
LEARNING_OBJS = $(LEARNING_SOURCES:.cpp=.o)
MNIST_OBJS = $(MNIST_SOURCES:.cpp=.o)

# Targets
TENSOR_TARGET = ./TensorTests
LEARNING_TARGET = ./Learning
MNIST_TARGET = ./MNIST

all: $(TENSOR_TARGET) $(LEARNING_TARGET) $(MNIST_TARGET)

# Build TensorTests
$(TENSOR_TARGET): $(TENSOR_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(TENSOR_OBJS)

# Build Learning
$(LEARNING_TARGET): $(LEARNING_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(LEARNING_OBJS)

# Build MNIST
$(MNIST_TARGET): $(MNIST_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(MNIST_OBJS)

# Rule to compile .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean up build files
clean:
	rm -f \
        $(TENSOR_TARGET) $(LEARNING_TARGET) $(MNIST_TARGET) \
        $(TENSOR_OBJS) $(LEARNING_OBJS) $(MNIST_OBJS)
