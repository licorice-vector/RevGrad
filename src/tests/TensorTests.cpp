#include <iostream>

#include "../tensor/Tensor.h"

using namespace RevGrad;

void unravel_index() {
    Shape shape({2, 3});
    if (shape.unravel_index(3) != std::vector<int>{1, 0}) {
        throw std::logic_error("unravel_index FAILED!");
    }
    std::cout << "unravel_index PASSED!" << std::endl;
}

void broadcastable() {
    Shape a({2, 3});
    Shape b({2, 3});
    Shape c({1, 3});
    Shape d({1, 4});
    if (!a.broadcastable(b) || !a.broadcastable(c) || c.broadcastable(d)) {
        throw std::logic_error("broadcastable FAILED!");
    }
    std::cout << "broadcastable PASSED!" << std::endl;
}

void broadcast_shape() {
    Shape a({2, 3});
    Shape b({2, 3});
    Shape c({1, 3});
    if (Shape::broadcast_shape(a, b) != a || Shape::broadcast_shape(a, c) != a) {
        throw std::logic_error("broadcast_shape FAILED!");
    }
    std::cout << "broadcast_shape PASSED!" << std::endl;
}

void get_values() {
    Tensor a(Shape({2}), 2);
    a.set({1}, 1);
    std::vector<float> values = a.get_values();
    if (values != std::vector<float>{2, 1}) {
        throw std::logic_error("get_values FAILED!");
    }
    std::cout << "get_values PASSED!" << std::endl;
}

void set() {
    Tensor a(Shape({2}), 2);
    a.set({1}, 1);
    if (a[{0}].value() != 2 || a[{1}].value() != 1) {
        throw std::logic_error("set FAILED!");
    }
    std::cout << "set PASSED!" << std::endl;
}

void flattened() {
    Tensor a(Shape({2, 3}), 5);
    Tensor b = a.flattened();
    if (a.shape != Shape({2, 3}) || b[{0}].value() != 5 || b.shape != Shape({6})) {
        throw std::logic_error("flattened FAILED!");
    }
    std::cout << "flattened PASSED!" << std::endl;
}

void reshaped() {
    Tensor a(Shape({2, 3}), 5);
    Tensor b = a.reshaped(Shape({3, 2}));
    if (a.shape != Shape({2, 3}) || b[{0, 0}].value() != 5 || b.shape != Shape({3, 2})) {
        throw std::logic_error("reshaped FAILED!");
    }
    std::cout << "reshaped PASSED!" << std::endl;
}

void transposed() {
    Tensor a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    Tensor b = a.transposed();
    if (
        a[{0, 1}].value() != 2 || 
        a.shape != Shape({2, 3}) || 
        b[{0, 1}].value() != 4 || 
        b.shape != Shape({3, 2})
    ) {
        throw std::logic_error("transposed FAILED!");
    }
    std::cout << "transposed PASSED!" << std::endl;
}

void backward_edges() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c = a + b;
    if (
        a[{0}].backward_edges().size() != 0 || 
        b[{0}].backward_edges().size() != 0 || 
        c[{0}].backward_edges().size() != 2
    ) {
        throw std::logic_error("backward_edges FAILED!");
    }
    std::cout << "backward_edges PASSED!" << std::endl;
}

void addition() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c = a + b;
    if (
        a[{0}].value() != 2 || 
        b[{0}].value() != 4 || 
        c[{0}].value() != 6
    ) {
        throw std::logic_error("addition FAILED!");
    }
    std::cout << "addition PASSED!" << std::endl;
}

void addition_gradient() {
    Tensor a(Shape({1}), 2);
    Tensor b(Shape({1}), 4);
    Tensor c = a + b;
    c.backward();
    if (
        a[{0}].grad() != 1 ||
        b[{0}].grad() != 1 ||
        c[{0}].grad() != 1
    ) {
        throw std::logic_error("addition_gradient FAILED!");
    }
    std::cout << "addition_gradient PASSED!" << std::endl;
}

void multiplication() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c = a * b;
    if (
        a[{0}].value() != 2 ||
        b[{0}].value() != 4 ||
        c[{0}].value() != 8
    ) {
        throw std::logic_error("multiplication FAILED!");
    }
    std::cout << "multiplication PASSED!" << std::endl;
}

void multiplication_gradient() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c = a * b;
    c.backward();
    if (
        a[{0}].grad() != 4 ||
        b[{0}].grad() != 2 ||
        c[{0}].grad() != 1
    ) {
        throw std::logic_error("multiplication_gradient FAILED!");
    }
    std::cout << "multiplication_gradient PASSED!" << std::endl;
}

void mixed_gradient() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c(Shape({2}), 6);
    Tensor d = c + a * b;
    d.backward();
    if (
        a[{0}].grad() != 4 ||
        b[{0}].grad() != 2 ||
        c[{0}].grad() != 1
    ) {
        throw std::logic_error("mixed_gradient FAILED!");
    }
    std::cout << "mixed_gradient PASSED!" << std::endl;
}

void large_addition() {
    int n = 100'000;
    std::vector<Tensor> tensors;
    for (int i = 0; i < n; i++) {
        tensors.push_back(Tensor(Shape({1}), 1));
    }
    Tensor x = tensors[0];
    for (int i = 1; i < n; i++) {
        x = x + tensors[i];
    }
    x.backward();
    std::cout << "large_addition PASSED!" << std::endl;
}

void matrix_matmul() {
    Tensor a(Shape({2, 3}), 2.0);
    Tensor b(Shape({3, 2}), 4.0);
    Tensor c = Tensor::matmul(a, b);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (c[{i, j}].value() != 24) {
                throw std::logic_error("matrix_matmul FAILED!");
            }
        }
    }
    std::cout << "matrix_matmul PASSED!" << std::endl;
}

void vector_matmul() {
    Tensor a(Shape({3}), 2);
    Tensor b(Shape({3}), 4);
    Tensor c = Tensor::matmul(a, b);
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 1; j++) {
            if (c[{i, j}].value() != 24) {
                throw std::logic_error("vector_matmul FAILED!");
            }
        }
    }
    std::cout << "vector_matmul PASSED!" << std::endl;
}

void exp() {
    Tensor a(Shape({2, 3}), 2);
    Tensor b = Tensor::exp(a);
    if (abs(b[{0, 0}].value() - 7.38905609893) > 0.001) {
        throw std::logic_error("exp FAILED!");
    }
    b.backward();
    std::cout << "exp PASSED!" << std::endl;
}

void relu() {
    Tensor a(Shape({2, 3}), -1.0);
    Tensor b = Tensor::relu(a);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (b[{i, j}].value() != 0) {
                throw std::logic_error("relu FAILED!");
            }
        }
    }
    b.backward();
    std::cout << "relu PASSED!" << std::endl;
}

void softmax() {
    Tensor a(Shape({2, 1}), 1);
    a.set({1, 0}, 3);
    Tensor b = Tensor::softmax(a);
    if (
        abs(b[{0, 0}].value() - 0.119203) > 0.001 || 
        abs(b[{1, 0}].value() - 0.119203) > 0.001) {
        throw std::logic_error("softmax FAILED!");
    }
    b.backward();
    std::cout << "softmax PASSED!" << std::endl;
}

void sigmoid() {
    Tensor a(Shape({2}), 1);
    Tensor b = Tensor::sigmoid(a);
    if (abs(b[{0}].value() - 0.731058578630074) > 0.001) {
        throw std::logic_error("sigmoid FAILED!");
    }
    b.backward();
    std::cout << "sigmoid PASSED!" << std::endl;
}

int main() {

    std::vector<void(*)()> tests = {
        &unravel_index,
        &broadcastable,
        &broadcast_shape,
        &get_values,
        &set,
        &flattened,
        &reshaped,
        &transposed,
        &backward_edges,
        &addition,
        &addition_gradient,
        &multiplication,
        &multiplication_gradient,
        &mixed_gradient,
        &large_addition,
        &matrix_matmul,
        &vector_matmul,
        &exp,
        &relu,
        &softmax,
        &sigmoid
    };
    for (auto test : tests) {
        test();
    }
    
    return 0;
}