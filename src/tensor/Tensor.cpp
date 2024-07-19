#include <iostream>
#include <cassert>

#include "Tensor.h"

namespace RevGrad {
    int Shape::size() const {
        return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>());
    }

    int Shape::dim() const {
        return dimensions.size();
    }

    int Shape::operator[](int idx) const {
        return dimensions[idx];
    }

    std::vector<int> Shape::unravel_index(int idx) const {
        std::vector<int> idxs(dim());
        int r = idx;
        for (int i = dim() - 1; i >= 0; i--) {
            idxs[i] = r % dimensions[i];
            r /= dimensions[i];
        }
        assert(r == 0); // index out of range
        return idxs;
    }

    bool Shape::operator==(const Shape& other) const {
        return dimensions == other.dimensions;
    }

    bool Shape::operator!=(const Shape& other) const {
        return dimensions != other.dimensions;
    }

    bool Shape::broadcastable(const Shape& other) const {
        int i = dim() - 1;
        int j = other.dim() - 1;
        while (i >= 0 && j >= 0) {
            if (dimensions[i] != other[j] && dimensions[i] != 1 && other[j] != 1) {
                return false;
            }
            i--, j--;
        }
        while (j >= 0) {
            if (other[j] != 1) {
                return false;
            }
            j--;
        }
        while (i >= 0) {
            if (dimensions[i] != 1) {
                return false;
            }
            i--;
        }
        return true;
    }

    Shape Shape::broadcast_shape(const Shape& a, const Shape& b) {
        int n = std::max(a.dim(), b.dim());
        std::vector<int> dimensions(n);
        int i = a.dim() - 1;
        int j = b.dim() - 1;
        int k = n - 1;
        while (i >= 0 && j >= 0) {
            dimensions[k] = std::max(a[i], b[j]);
            i--, j--, k--;
        }
        while (i >= 0) {
            dimensions[k] = a[i];
            i--, k--;
        }
        while (j >= 0) {
            dimensions[k] = b[j];
            j--, k--;
        }
        return Shape(dimensions.begin(), dimensions.end());
    }

    std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "(";
        for (int i = 0; i < shape.dim(); i++) {
            os << shape[i];
            if (i < shape.dim() - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }

    std::random_device Tensor::rd = std::random_device();
    std::mt19937 Tensor::rng = std::mt19937(rd());

    std::vector<float> Tensor::random_vector(int n) {
        std::vector<float> r(n);
        std::uniform_real_distribution<float> uniform_dist(0.0f, 0.1f);
        for (int i = 0; i < n; i++) {
            r[i] = uniform_dist(rng);
        }
        return r;
    }

    std::vector<float> Tensor::random_vector(int n, int fan_in) {
        std::vector<float> r(n);
        std::normal_distribution<float> he_dist(0.0f, std::sqrt(2.0f / fan_in));
        for (int i = 0; i < n; i++) {
            r[i] = he_dist(rng);
        }
        return r;
    }

    Tensor::Tensor() {
        this->shape = Shape({0});
    }

    Tensor::Tensor(Shape shape, float init_value) {
        this->shape = shape;
        for (int i = 0; i < shape.size(); i++) {
            this->data.push_back(Float(init_value));
        }
    }

    Tensor::Tensor(Shape shape, std::vector<float> values) {
        assert(shape.size() == values.size());
        this->shape = shape;
        for (int i = 0; i < shape.size(); i++) {
            this->data.push_back(Float(values[i]));
        }
    }

    Tensor::Tensor(const Tensor& other) {
        this->data = other.data;
        this->shape = other.shape;
    }

    Tensor::Tensor(const Float& float_data) {
        this->shape = Shape({1});
        this->data.push_back(float_data);
    }

    Tensor Tensor::random(Shape shape) {
        return Tensor(shape, random_vector(shape.size()));
    }

    Tensor Tensor::random(Shape shape, int fan_in) {
        return Tensor(shape, random_vector(shape.size(), fan_in));
    }

    const Float Tensor::operator[](std::vector<int> idxs) const {
        int d = shape.dim();
        assert(idxs.size() == d);
        int flat_idx = 0, mult = 1, dim = d;
        for (int i = dim - 1; i >= 0; i--) {
            flat_idx += idxs[i] * mult;
            mult *= shape[i];
        }
        return this->data[flat_idx];
    }

    std::vector<float> Tensor::get_values() const {
        std::vector<float> values;
        for (int i = 0; i < this->shape.size(); i++) {
            values.push_back(this->data[i].value());
        }
        return values;
    }

    void Tensor::set(std::vector<int> idxs, float new_value) {
        int d = shape.dim();
        assert(idxs.size() == d);
        int flat_idx = 0, mult = 1, dim = d;
        for (int i = dim - 1; i >= 0; i--) {
            flat_idx += *(idxs.begin() + i) * mult;
            mult *= shape[i];
        }
        this->data[flat_idx] = Float(new_value);
    }

    Tensor Tensor::flattened() {
        return reshaped(Shape({this->shape.size()}));
    }
    
    Tensor Tensor::reshaped(const Shape& shape) {
        assert(this->shape.size() == shape.size());
        Tensor x = *this;
        x.shape = shape;
        return x;
    }

    Tensor Tensor::transposed() const {
        assert(this->shape.dim() == 2);
        int rows = this->shape[0];
        int cols = this->shape[1];
        Tensor transposed(Shape({cols, rows}), 0.0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.data[j * rows + i] = this->data[i * cols + j];
            }
        }
        return transposed;
    }

    Tensor operator+(const Tensor& a, const Tensor& b) {
        Shape shape = Shape::broadcast_shape(a.shape, b.shape);
        assert(shape.broadcastable(a.shape));
        assert(shape.broadcastable(b.shape));
        Tensor c(shape);
        for (int i = 0; i < c.shape.size(); i++) {
            c.data[i] = a.data[i % a.shape.size()] + b.data[i % b.shape.size()];
        }
        return c;
    }

    Tensor operator*(const Tensor& a, const Tensor& b) {
        Shape shape = Shape::broadcast_shape(a.shape, b.shape);
        assert(shape.broadcastable(a.shape));
        assert(shape.broadcastable(b.shape));
        Tensor c(shape);
        for (int i = 0; i < c.shape.size(); i++) {
            c.data[i] = a.data[i % a.shape.size()] * b.data[i % b.shape.size()];
        }
        return c;
    }

    Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
        Shape shape_a = a.shape;
        Shape shape_b = b.shape;
        if (shape_a.dim() == 1) {
            shape_a = Shape({1, shape_a[0]});
        }
        if (shape_b.dim() == 1) {
            shape_b = Shape({shape_b[0], 1});
        }
        assert(shape_a.dim() == 2 && shape_b.dim() == 2);
        assert(shape_a[1] == shape_b[0]);
        int rows_a = shape_a[0];
        int cols_a = shape_a[1];
        int cols_b = shape_b[1];
        Tensor c(Shape({rows_a, cols_b}), 0.0);
        for (int i = 0; i < rows_a; i++) {
            for (int j = 0; j < cols_b; j++) {
                Float sum(0.0f);
                for (int k = 0; k < cols_a; k++) {
                    sum = sum + a.data[i * cols_a + k] * b.data[k * cols_b + j];
                }
                c.data[i * cols_b + j] = sum;
            }
        }
        return c;
    }

    Tensor Tensor::exp(const Tensor& x) {
        int n = x.shape.size();
        Tensor y(x.shape, 0.0);
        for (int i = 0; i < n; i++) {
            y.data[i] = Float::exp(x.data[i]);
        }
        return y;
    }

    Tensor Tensor::relu(const Tensor& x) {
        int n = x.shape.size();
        Tensor y(x.shape, 0.0);
        for (int i = 0; i < n; i++) {
            y.data[i] = Float::relu(x.data[i]);
        }
        return y;
    }

    Tensor Tensor::softmax(const Tensor& x) {
        assert(x.shape.dim() == 2); // ensure shape is (features, batch_size)
        int features = x.shape[0];
        int batch_size = x.shape[1];
        Tensor y(x.shape, 0.0);
        for (int b = 0; b < batch_size; b++) {
            std::vector<Float> exp_data(features);
            Float exp_sum(0.0f);
            for (int f = 0; f < features; f++) {
                exp_data[f] = Float::exp(x.data[f * batch_size + b]);
                exp_sum = exp_sum + exp_data[f];
            }
            for (int f = 0; f < features; f++) {
                y.data[f * batch_size + b] = exp_data[f] / exp_sum;
            }
        }
        return y;
    }

    Tensor Tensor::sigmoid(const Tensor& x) {
        int n = x.shape.size();
        Tensor y(x.shape, 0.0);
        for (int i = 0; i < n; i++) {
            Float data = x.data[i];
            if (0 < data.value()) {
                y.data[i] = Float(1.0) / (Float(1.0) + Float::exp(-data));
            } else {
                Float exp_val = Float::exp(data);
                y.data[i] = exp_val / (Float(1.0) + exp_val);
            }
        }
        return y;
    }

    void Tensor::backward(const std::vector<float>& gradient) {
        int n = shape.size();
        for (int i = 0; i < n; i++) {
            data[i].backward(gradient[i]);
        }
    }

    void Tensor::backward() {
        this->backward(std::vector<float>(this->shape.size(), 1));
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& x) {
        os << "Tensor(shape=" << x.shape << ", data=";
        const std::vector<float> values = x.get_values();
        int dimensions = x.shape.dim();
        std::function<void(int, int)> print_tensor = [&] (int dim, int offset) {
            if (dim == dimensions - 1) {
                os << "[";
                for (int i = 0; i < x.shape[dim]; i++) {
                    os << values[offset + i];
                    if (i < x.shape[dim] - 1) {
                        os << ", ";
                    }
                }
                os << "]";
            } else {
                os << "[";
                for (int i = 0; i < x.shape[dim]; i++) {
                    if (0 < i) {
                        os << ", ";
                        for (int j = 0; j <= dim; j++) {
                            os << " ";
                        }
                    }
                    print_tensor(dim + 1, offset + i * x.shape.size() / x.shape[dim]);
                }
                os << "])";
            }
        };
        print_tensor(0, 0);
        return os;
    }
}
