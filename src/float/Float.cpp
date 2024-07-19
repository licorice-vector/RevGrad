#include <iostream>
#include <cassert>

#include "Float.h"

namespace RevGrad {
    std::vector<std::shared_ptr<FloatData>> Float::memory = std::vector<std::shared_ptr<FloatData>>();

    Float::Float(float init_value) {
        this->ptr = std::make_shared<FloatData>(init_value);
        Float::memory.push_back(this->ptr);
    }

    float Float::value() const {
        return this->ptr->value;
    }

    float Float::grad() const {
        return this->ptr->grad;
    }

    int Float::unresolved() const {
        return this->ptr->unresolved;
    }

    std::vector<BackwardEdge> Float::backward_edges() const {
        return this->ptr->backward_edges;
    }

    void Float::set_value(float value) const {
        this->ptr->value = value;
    }

    void Float::set_grad(float grad) const {
        this->ptr->grad = grad;
    }
    
    void Float::add_backward_edge(BackwardEdge edge) const {
        this->ptr->backward_edges.push_back(edge);
    }

    void Float::increase_unresolved() const {
        this->ptr->unresolved++;
    }

    void Float::decrease_unresolved() const {
        this->ptr->unresolved = std::max<int>(this->ptr->unresolved - 1, 0);
    }
    
    Float operator+(const Float& a, const Float& b) {
        Float c(a.value() + b.value());
        // derivative of a + b w.r.t a
        c.add_backward_edge({a, 1.0});
        // derivative of a + b w.r.t b
        c.add_backward_edge({b, 1.0});
        a.increase_unresolved();
        b.increase_unresolved();
        return c;
    }

    Float operator*(const Float& a, const Float& b) {
        Float c(a.value() * b.value());
        // derivative of a * b w.r.t a
        c.add_backward_edge({a, b.value()});
        // derivative of a * b w.r.t b
        c.add_backward_edge({b, a.value()});
        a.increase_unresolved();
        b.increase_unresolved();
        return c;
    }

    Float operator-(const Float& a, const Float& b) {
        Float c(a.value() - b.value());
        // derivative of a - b w.r.t a
        c.add_backward_edge({a, 1.0});
        // derivative of a - b w.r.t b
        c.add_backward_edge({b, -1.0});
        a.increase_unresolved();
        b.increase_unresolved();
        return c;
    }

    Float operator/(const Float& a, const Float& b) {
        Float c(a.value() / b.value());
        // derivative of a / b w.r.t a
        c.add_backward_edge({a, 1.0f / b.value()});
        // derivative of a / b w.r.t b
        c.add_backward_edge({b, -a.value() / (b.value() * b.value())});
        a.increase_unresolved();
        b.increase_unresolved();
        return c;
    }

    Float Float::operator-() const {
        return Float(0.0f) - *this;
    }

    bool Float::operator==(const Float& other) const {
        return ptr == other.ptr;
    }

    bool Float::operator!=(const Float& other) const {
        return !(*this == other);
    }

    Float Float::apply_unary_operation(
        std::function<float(float)> op, 
        std::function<float(float)> grad_op
    ) const {
        float value = this->value();
        Float x(op(value));
        x.add_backward_edge({*this, grad_op(value)});
        this->increase_unresolved();
        return x;
    }

    Float Float::exp(const Float& x) {
        return x.apply_unary_operation(
            [] (float x) { return std::exp(x); },
            [] (float x) { return std::exp(x); }
        );
    }

    Float Float::relu(const Float& x) {
        return x.apply_unary_operation(
            [] (float x) -> float { return std::max<float>(0.0, x); },
            [] (float x) -> float { return x > 0 ? 1.0 : 0.0; }
        );
    }

    void Float::clear_memory() {
        Float::memory.clear();
    }

    void Float::backward(const float& gradient) {
        std::stack<BackwardEdge> S;
        S.push({*this, gradient});
        while (!S.empty()) {
            auto [node, grad] = S.top();
            S.pop();
            node.set_grad(node.grad() + grad);
            node.decrease_unresolved();
            if (node.unresolved() <= 0) {
                for (auto edge : node.backward_edges()) {
                    S.push({edge.node, node.grad() * edge.grad});
                }
            }
        }
    }

    void Float::backward() {
        this->backward(1);
    }

    std::ostream& operator<<(std::ostream& os, const Float& x) {
        os << "(ptr: " << x.ptr;
        os << ", value: " << x.value();
        os << ", grad: " << x.grad();
        os << ", unresolved: " << x.unresolved();
        os << ", backward_edges: [";
        for (size_t i = 0; i < x.ptr->backward_edges.size(); i++) {
            auto& edge = x.ptr->backward_edges[i];
            os << "(ptr: " << edge.node.ptr << ", grad: " << edge.grad << ")";
            if (i + 1 != x.ptr->backward_edges.size()) {
                os << ", ";
            }
        }
        os << "])";
        return os;
    }

    BackwardEdge::BackwardEdge(Float node, float grad) : node(node), grad(grad) {}
}
