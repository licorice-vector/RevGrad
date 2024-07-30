#include <iostream>

#include "../utill/Print.h"
#include "../tensor/Tensor.h"

using namespace RevGrad;

void broadcast_shape() {
    Shape a({2, 3});
    Shape b({2, 3});
    Shape c({1, 3});
    if (ViewUtill::broadcast_shape(a, b) != a || ViewUtill::broadcast_shape(a, c) != a) {
        throw std::logic_error("broadcast_shape FAILED!");
    }
    std::cout << "broadcast_shape PASSED!" << std::endl;
}

void set() {
    Tensor a(Shape({2}), 2);
    a.value({1}) = 1;
    if (a.value({0}) != 2 || a.value({1}) != 1) {
        throw std::logic_error("set FAILED!");
    }
    std::cout << "set PASSED!" << std::endl;
}

void flatten() {
    Tensor a(Shape({2, 3}), 5);
    a.flatten();
    if (a.value({0}) != 5 || a.shape() != Shape({6})) {
        throw std::logic_error("flatten FAILED!");
    }
    std::cout << "flatten PASSED!" << std::endl;
}

void reshape() {
    Tensor a(Shape({2, 3}), 5);
    a.reshape(Shape({3, 2}));
    if (a.value({0, 0}) != 5 || a.shape() != Shape({3, 2})) {
        throw std::logic_error("reshape FAILED!");
    }
    std::cout << "reshape PASSED!" << std::endl;
}

void transpose() {
    Tensor a(Shape({2, 3}), {1, 2, 3, 4, 5, 6});
    a.transpose();
    if (
        a.value({0, 1}) != 4 || 
        a.shape() != Shape({3, 2})
    ) {
        throw std::logic_error("transpose FAILED!");
    }
    std::cout << "transpose PASSED!" << std::endl;
}

void slice() {
    Tensor a = Tensor(Shape({5, 5}), {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    });
    a.transpose();
    Tensor b = a.slice({{2, 4}, {1, 3}});
    if (
        b.size() != 4 ||
        b.value({0, 0}) != 8 || 
        b.value({1, 0}) != 9 || 
        b.value({0, 1}) != 13 || 
        b.value({1, 1}) != 14 || 
        b.shape() != Shape({2, 2})
    ) {
        throw std::logic_error("slice FAILED!");
    }
    std::cout << "slice PASSED!" << std::endl;
}

void edges() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c = a + b;
    if (
        a.edges().size() != 0 || 
        b.edges().size() != 0 || 
        c.edges().size() != 2
    ) {
        throw std::logic_error("edges FAILED!");
    }
    std::cout << "edges PASSED!" << std::endl;
}

void addition() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c = a + b;
    if (
        a.value({0}) != 2 || 
        b.value({0}) != 4 || 
        c.value({0}) != 6
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
        a.grad({0}) != 1 ||
        b.grad({0}) != 1 ||
        c.grad({0}) != 1
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
        a.value({0}) != 2 ||
        b.value({0}) != 4 ||
        c.value({0}) != 8
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
        a.grad({0}) != 4 ||
        b.grad({0}) != 2 ||
        c.grad({0}) != 1
    ) {
        throw std::logic_error("multiplication_gradient FAILED!");
    }
    std::cout << "multiplication_gradient PASSED!" << std::endl;
}

void division() {
    Tensor a(Shape({2}), 4);
    Tensor b(Shape({2}), 2);
    Tensor c = a / b;
    if (
        a.value({0}) != 4 ||
        b.value({0}) != 2 ||
        c.value({0}) != 2
    ) {
        throw std::logic_error("division FAILED!");
    }
    std::cout << "division PASSED!" << std::endl;
}

void division_gradient() {
    Tensor a(Shape({2}), 4);
    Tensor b(Shape({2}), 2);
    Tensor c = a / b;
    c.backward();
    if (
        a.grad({0}) != 0.5 ||
        b.grad({0}) != -1.0 ||
        c.grad({0}) != 1
    ) {
        throw std::logic_error("division_gradient FAILED!");
    }
    std::cout << "division_gradient PASSED!" << std::endl;
}

void mixed_gradient() {
    Tensor a(Shape({2}), 2);
    Tensor b(Shape({2}), 4);
    Tensor c(Shape({2}), 6);
    Tensor d = c + a * b;
    d.backward();
    if (
        a.grad({0}) != 4 ||
        b.grad({0}) != 2 ||
        c.grad({0}) != 1
    ) {
        throw std::logic_error("mixed_gradient FAILED!");
    }
    std::cout << "mixed_gradient PASSED!" << std::endl;
}

void large_addition() {
    int n = 10'000;
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

void sum() {
    Tensor a(Shape({50}), 1);
    Tensor b = Tensor::sum(a);
    b.backward();
    if (b.value({0}) != 50 || b.grad({0}) != 1 || b.size() != 1) {
        throw std::logic_error("sum FAILED!");
    }
    std::cout << "sum PASSED!" << std::endl;
}

void max() {
    Tensor a(Shape({2, 2}), 1);
    a.value({0, 0}) = 2;
    a.value({0, 1}) = 4;
    a.value({1, 0}) = 1;
    a.value({1, 1}) = 5;
    Tensor b = Tensor::max(a);
    b.backward();
    if (b.value({0}) != 2 || b.value({1}) != 5 || b.size() != 2) {
        throw std::logic_error("max FAILED!");
    }
    if (a.grad({0, 0}) != 1 || a.grad({0, 1}) != 0 || a.grad({1, 0}) != 0 || a.grad({1, 1}) != 1) {
        throw std::logic_error("max FAILED!");
    }
    Tensor x(Shape({2, 3}), {0, 1, 4, 0, 7, 1});
    Tensor y = Tensor::max(x);
    y.backward();
    if (x.grads() != std::vector<float>{0.5, 0, 1, 0.5, 1, 0}) {
        throw std::logic_error("max FAILED!");
    }
    std::cout << "max PASSED!" << std::endl;
}

void exp() {
    Tensor a(Shape({2, 3}), 2);
    Tensor b = Tensor::exp(a);
    b.backward();
    if (
        abs(b.value({0, 0}) - 7.38905609893) > 0.001 || 
        abs(a.grad({0, 0}) - 7.38905609893) > 0.001
    ) {
        throw std::logic_error("exp FAILED!");
    }
    std::cout << "exp PASSED!" << std::endl;
}

void relu() {
    Tensor a(Shape({2, 3}), -1.0);
    Tensor b = Tensor::relu(a);
    b.backward();
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (b.value({i, j}) != 0 || a.grad({i, j}) != 0) {
                throw std::logic_error("relu FAILED!");
            }
        }
    }
    std::cout << "relu PASSED!" << std::endl;
}

void softmax() {
    Tensor a(Shape({2, 1}), 1.0);
    a.value({1, 0}) = 3.0;
    Tensor b = Tensor::softmax(a);
    b.backward();
    if (
        abs(b.value({0, 0}) - (0.119203)) > 0.0001 || 
        abs(b.value({1, 0}) - (0.880797)) > 0.0001 ||
        abs(a.grad({0, 0})) > 0.0001 || 
        abs(a.grad({1, 0})) > 0.0001
    ) {
        throw std::logic_error("softmax FAILED!");
    }
    std::cout << "softmax PASSED!" << std::endl;
}

void log_softmax() {
    Tensor a(Shape({2, 1}), 1);
    a.value({1, 0}) = 3;
    Tensor b = Tensor::log_softmax(a);
    b.backward();
    if (
        abs(b.value({0, 0}) - (-2.126927)) > 0.0001 || 
        abs(b.value({1, 0}) - (-0.126928)) > 0.0001 ||
        abs(a.grad({0, 0}) - (0.761594)) > 0.0001 || 
        abs(a.grad({1, 0}) - (-0.761594)) > 0.0001
    ) {
        throw std::logic_error("log_softmax FAILED!");
    }
    std::cout << "log_softmax PASSED!" << std::endl;
}

void sigmoid() {
    Tensor a(Shape({2}), 0.4);
    Tensor b = Tensor::sigmoid(a);
    b.backward();
    if (
        abs(b.value({0}) - 0.598687) > 0.001 ||
        abs(a.grad({0}) - 0.240260) > 0.001
    ) {
        throw std::logic_error("sigmoid FAILED!");
    }
    std::cout << "sigmoid PASSED!" << std::endl;
}

void matmul() {
    Tensor a(Shape({2, 3}), 2.0);
    Tensor b(Shape({3, 2}), 4.0);
    Tensor c = Tensor::matmul(a, b);
    c.backward();
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (c.value({i, j}) != 24 || a.grad({0}) != 8 || b.grad({0}) != 4) {
                throw std::logic_error("matmul FAILED!");
            }
        }
    }
    std::cout << "matmul PASSED!" << std::endl;
}

void matmul_gradient() {
    Tensor a(Shape({3, 2}), {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b(Shape({2, 3}), {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    Tensor c = Tensor::matmul(a, b);
    c.backward();
    if (
        c.values() != Values{27,  30,  33, 61, 68, 75, 95, 106, 117} ||
        a.grads() != Gradients{24, 33, 24, 33, 24, 33} ||
        b.grads() != Gradients{9, 9, 9, 12, 12, 12}
    ) {
        throw std::logic_error("matmul_gradient FAILED!");
    }
    std::cout << "matmul_gradient PASSED!" << std::endl;
}

int main() {

    std::vector<void(*)()> tests = {
        &broadcast_shape,
        &set,
        &flatten,
        &reshape,
        &transpose,
        &slice,
        &edges,
        &addition,
        &addition_gradient,
        &multiplication,
        &multiplication_gradient,
        &division,
        &division_gradient,
        &mixed_gradient,
        &large_addition,
        &sum,
        &max,
        &exp,
        &relu,
        &softmax,
        &log_softmax,
        &sigmoid,
        &matmul,
        &matmul_gradient
    };
    for (auto test : tests) {
        test();
    }
    
    return 0;
}
