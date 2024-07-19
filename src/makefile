CXX = g++-13
CXXFLAGS = -std=c++17 -g
LDFLAGS = -Wl,-ld_classic

# Source files for each target
FLOAT_SOURCES = \
	./float/Float.cpp \
	./tests/FloatTests.cpp

TENSOR_SOURCES = \
    ./float/Float.cpp \
    ./tensor/Tensor.cpp \
    ./tests/TensorTests.cpp

LEARNING_SOURCES = \
    ./float/Float.cpp \
    ./tensor/Tensor.cpp \
    ./loss/Loss.cpp \
    ./strategy/Strategy.cpp \
    ./model/Model.cpp \
    ./examples/Learning.cpp

# Object files for each target
FLOAT_OBJS = $(FLOAT_SOURCES:.cpp=.o)
TENSOR_OBJS = $(TENSOR_SOURCES:.cpp=.o)
LEARNING_OBJS = $(LEARNING_SOURCES:.cpp=.o)

# Targets
FLOAT_TARGET = ./FloatTests
TENSOR_TARGET = ./TensorTests
LEARNING_TARGET = ./Learning

all: $(FLOAT_TARGET) $(TENSOR_TARGET) $(LEARNING_TARGET)

# Build FloatTests
$(FLOAT_TARGET): $(FLOAT_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(FLOAT_OBJS)

# Build TensorTests
$(TENSOR_TARGET): $(TENSOR_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(TENSOR_OBJS)

# Build Learning
$(LEARNING_TARGET): $(LEARNING_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(LEARNING_OBJS)

# Rule to compile .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean up build files
clean:
	rm -f $(FLOAT_TARGET) $(TENSOR_TARGET) $(LEARNING_TARGET) $(FLOAT_OBJS) $(TENSOR_OBJS) $(LEARNING_OBJS)
