#include <iostream>

#include "../float/Float.h"

using namespace RevGrad;

void addition_value() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a + b;
    if (
        a.value() != 2 || 
        b.value() != 4 || 
        c.value() != 6
    ) {
        throw std::logic_error("addition_value FAILED!");
    }
    std::cout << "addition_value PASSED!" << std::endl;
}

void addition_grad() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a + b;
    c.backward();
    if (
        a.grad() != 1 || 
        b.grad() != 1 || 
        c.grad() != 1
    ) {
        throw std::logic_error("addition_grad FAILED!");
    }
    std::cout << "addition_grad PASSED!" << std::endl;
}

void addition_backward_edges() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a + b;
    if (
        a.backward_edges().size() != 0 || 
        b.backward_edges().size() != 0 || 
        c.backward_edges().size() != 2
    ) {
        throw std::logic_error("addition_backward_edges FAILED!");
    }
    std::cout << "addition_backward_edges PASSED!" << std::endl;
}

void multiplication_value() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a * b;
    if (
        a.value() != 2 || 
        b.value() != 4 || 
        c.value() != 8
    ) {
        throw std::logic_error("multiplication_value FAILED!");
    }
    std::cout << "multiplication_value PASSED!" << std::endl;
}

void multiplication_grad() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a * b;
    c.backward();
    if (
        a.grad() != 4 || 
        b.grad() != 2 || 
        c.grad() != 1
    ) {
        throw std::logic_error("multiplication_grad FAILED!");
    }
    std::cout << "multiplication_grad PASSED!" << std::endl;
}

void multiplication_backward_edges() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a * b;
    if (
        a.backward_edges().size() != 0 || 
        b.backward_edges().size() != 0 || 
        c.backward_edges().size() != 2
    ) {
        throw std::logic_error("multiplication_backward_edges FAILED!");
    }
    std::cout << "multiplication_backward_edges PASSED!" << std::endl;
}

void division_value() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a / b;
    if (
        a.value() != 2 || 
        b.value() != 4 || 
        c.value() != 0.5
    ) {
        throw std::logic_error("division_value FAILED!");
    }
    std::cout << "division_value PASSED!" << std::endl;
}

void division_grad() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a / b;
    c.backward();
    if (
        a.grad() != 0.25 || 
        b.grad() != -0.125 || 
        c.grad() != 1
    ) {
        throw std::logic_error("division_grad FAILED!");
    }
    std::cout << "division_grad PASSED!" << std::endl;
}

void division_backward_edges() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a / b;
    if (
        a.backward_edges().size() != 0 || 
        b.backward_edges().size() != 0 || 
        c.backward_edges().size() != 2
    ) {
        throw std::logic_error("division_backward_edges FAILED!");
    }
    std::cout << "division_backward_edges PASSED!" << std::endl;
}

void subtraction_value() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a - b;
    if (
        a.value() != 2 || 
        b.value() != 4 || 
        c.value() != -2
    ) {
        throw std::logic_error("subtraction_value FAILED!");
    }
    std::cout << "subtraction_value PASSED!" << std::endl;
}

void subtraction_grad() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a - b;
    c.backward();
    if (
        a.grad() != 1 || 
        b.grad() != -1 || 
        c.grad() != 1
    ) {
        throw std::logic_error("subtraction_grad FAILED!");
    }
    std::cout << "subtraction_grad PASSED!" << std::endl;
}

void subtraction_backward_edges() {
    Float a(2.0f);
    Float b(4.0f);
    Float c = a - b;
    if (
        a.backward_edges().size() != 0 || 
        b.backward_edges().size() != 0 || 
        c.backward_edges().size() != 2
    ) {
        throw std::logic_error("subtraction_backward_edges FAILED!");
    }
    std::cout << "subtraction_backward_edges PASSED!" << std::endl;
}

void exp_value() {
    Float a(2.0f);
    Float b = Float::exp(a);
    if (
        a.value() != 2 || 
        abs(b.value() - 7.38905609893) > 0.001
    ) {
        throw std::logic_error("exp_value FAILED!");
    }
    std::cout << "exp_value PASSED!" << std::endl;
}

void exp_grad() {
    Float a(2.0f);
    Float b = Float::exp(a);
    b.backward();
    if (
        abs(a.grad() - 7.38905609893) > 0.001 || 
        b.grad() != 1
    ) {
        throw std::logic_error("exp_grad FAILED!");
    }
    std::cout << "exp_grad PASSED!" << std::endl;
}

void exp_backward_edges() {
    Float a(2.0f);
    Float b = Float::exp(a);
    if (
        a.backward_edges().size() != 0 || 
        b.backward_edges().size() != 1
    ) {
        throw std::logic_error("exp_backward_edges FAILED!");
    }
    std::cout << "exp_backward_edges PASSED!" << std::endl;
}

void relu_value() {
    Float a(2.0f);
    Float b = Float::relu(a);
    if (
        a.value() != 2 || 
        b.value() != 2
    ) {
        throw std::logic_error("relu_value FAILED!");
    }
    Float c(-2.0f);
    Float d = Float::relu(c);
    if (
        c.value() != -2 || 
        d.value() != 0
    ) {
        throw std::logic_error("relu_value FAILED!");
    }
    std::cout << "relu_value PASSED!" << std::endl;
}

void relu_grad() {
    Float a(2.0f);
    Float b = Float::relu(a);
    b.backward();
    if (
        a.grad() != 1 || 
        b.grad() != 1
    ) {
        throw std::logic_error("relu_grad FAILED!");
    }
    Float c(-2.0f);
    Float d = Float::relu(c);
    d.backward();
    if (
        c.grad() != 0 || 
        d.grad() != 1
    ) {
        throw std::logic_error("relu_grad FAILED!");
    }
    std::cout << "relu_grad PASSED!" << std::endl;
}

void relu_backward_edges() {
    Float a(2.0f);
    Float b = Float::relu(a);
    if (
        a.backward_edges().size() != 0 || 
        b.backward_edges().size() != 1
    ) {
        throw std::logic_error("relu_backward_edges FAILED!");
    }
    Float c(-2.0f);
    Float d = Float::relu(c);
    if (
        c.backward_edges().size() != 0 || 
        d.backward_edges().size() != 1
    ) {
        throw std::logic_error("relu_backward_edges FAILED!");
    }
    std::cout << "relu_backward_edges PASSED!" << std::endl;
}

void mixed_value() {
    Float a(2.0f);
    Float b(4.0f);
    Float c(6.0f);
    Float d = a + b * c;
    if (
        a.value() != 2 || 
        b.value() != 4 || 
        c.value() != 6 ||
        d.value() != 26
    ) {
        throw std::logic_error("mixed_value FAILED!");
    }
    std::cout << "mixed_value PASSED!" << std::endl;
}

void mixed_grad() {
    Float a(2.0f);
    Float b(4.0f);
    Float c(6.0f);
    Float d = a + b * c;
    d.backward();
    if (
        a.grad() != 1 || 
        b.grad() != 6 || 
        c.grad() != 4 ||
        d.grad() != 1
    ) {
        throw std::logic_error("mixed_grad FAILED!");
    }
    std::cout << "mixed_grad PASSED!" << std::endl;
}

void mixed_backward_edges() {
    Float a(2.0f);
    Float b(4.0f);
    Float c(6.0f);
    Float d = a + b * c;
    if (
        a.backward_edges().size() != 0 || 
        b.backward_edges().size() != 0 || 
        c.backward_edges().size() != 0 ||
        d.backward_edges().size() != 2
    ) {
        throw std::logic_error("mixed_backward_edges FAILED!");
    }
    std::cout << "mixed_backward_edges PASSED!" << std::endl;
}

int main() {

    std::vector<void(*)()> tests = {
        &addition_value,
        &addition_grad,
        &addition_backward_edges,
        &multiplication_value,
        &multiplication_grad,
        &multiplication_backward_edges,
        &division_value,
        &division_grad,
        &division_backward_edges,
        &subtraction_value,
        &subtraction_grad,
        &subtraction_backward_edges,
        &exp_value,
        &exp_grad,
        &exp_backward_edges,
        &relu_value,
        &relu_grad,
        &relu_backward_edges,
        &mixed_value,
        &mixed_grad,
        &mixed_backward_edges
    };
    for (auto test : tests) {
        test();
    }
    
    return 0;
}