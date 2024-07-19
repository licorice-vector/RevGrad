#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <numeric>
#include <memory>
#include <stack>
#include <functional>
#include <random>

#include "../float/Float.h"

namespace RevGrad {
    class Shape {
        std::vector<int> dimensions;
    public:
        template<typename... Args>
        Shape(Args... args) : dimensions{args...} {}
        int size() const;
        int dim() const;
        int operator[](int index) const;
        std::vector<int> unravel_index(int idx) const;
        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;
        /*
            Broadcasting is possible if:
            1. Dimensions match from the right to the left.
            2. Any dimension of size 1 can be broadcast to match the corresponding dimension in the other shape.
        */
        bool broadcastable(const Shape& other) const;
        static Shape broadcast_shape(const Shape& a, const Shape& b);
    };

    std::ostream& operator<<(std::ostream& os, const Shape& shape);

    class Tensor {
        static std::random_device rd;
        static std::mt19937 rng;
        static std::vector<float> random_vector(int n);
        static std::vector<float> random_vector(int n, int fan_in);
    public:
        Shape shape;
        std::vector<Float> data;
        Tensor();
        Tensor(Shape shape, float init_value = 0.0);
        Tensor(Shape shape, std::vector<float> values);
        Tensor(const Tensor& other);
        Tensor(const Float& float_data);
        static Tensor random(Shape shape);
        static Tensor random(Shape shape, int fan_in);
        const Float operator[](std::vector<int> idxs) const;
        std::vector<float> get_values() const;
        void set(std::vector<int> idxs, float new_value);
        Tensor flattened();
        Tensor reshaped(const Shape& shape);
        Tensor transposed() const;
        friend Tensor operator+(const Tensor& a, const Tensor& b);
        friend Tensor operator*(const Tensor& a, const Tensor& b);
        static Tensor matmul(const Tensor& a, const Tensor& b);
        static Tensor exp(const Tensor& x);
        static Tensor relu(const Tensor& x);
        static Tensor softmax(const Tensor& x);
        static Tensor sigmoid(const Tensor& x);
        /*
            Calls backward on children once its final gradient is computed
            @param gradient
                derivative of initial tensor (typically the loss) w.r.t caller
                multiplied by the derivative of the caller with respect to this
        */
        void backward(const std::vector<float>& gradient);
        void backward();
    };

    std::ostream& operator<<(std::ostream& os, const Tensor& x);
}

#endif
