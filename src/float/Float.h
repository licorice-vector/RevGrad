#ifndef FLOAT_H
#define FLOAT_H

#include <vector>
#include <functional>
#include <stack>
#include <cmath>
#include <memory>

namespace RevGrad {

    struct BackwardEdge;

    struct FloatData {
        float value;
        float grad = 0;
        int unresolved = 0;
        std::vector<BackwardEdge> backward_edges;
        FloatData(float value) : value(value) {}
    };

    class Float {
        // To avoid recursion limit when destroying shared_ptrs
        static std::vector<std::shared_ptr<FloatData>> memory;
        std::shared_ptr<FloatData> ptr;
    public:
        Float(float init_value = 0.0);
        float value() const;
        float grad() const;
        int unresolved() const;
        std::vector<BackwardEdge> backward_edges() const;
        void set_value(float value) const;
        void set_grad(float grad) const;
        void add_backward_edge(BackwardEdge edge) const;
        void increase_unresolved() const;
        void decrease_unresolved() const;
        friend Float operator+(const Float& a, const Float& b);
        friend Float operator*(const Float& a, const Float& b);
        friend Float operator-(const Float& a, const Float& b);
        friend Float operator/(const Float& a, const Float& b);
        friend std::ostream& operator<<(std::ostream& os, const Float& x);
        Float operator-() const;
        bool operator==(const Float& other) const;
        bool operator!=(const Float& other) const;
        Float apply_unary_operation(std::function<float(float)> op, std::function<float(float)> grad_op) const;
        static Float exp(const Float& x);
        static Float relu(const Float& x);
        /*
            Only call this if you know what you are doing
        */
        static void clear_memory();
        /*
            Calls backward on children once its final gradient is computed
            @param gradient
                derivative of initial tensor (typically the loss) w.r.t caller
                multiplied by the derivative of the caller with respect to this
        */
        void backward(const float& gradient);
        void backward();
    };

    struct BackwardEdge {
        Float node;
        float grad;
        BackwardEdge(Float node, float grad);
    };
}

#endif
