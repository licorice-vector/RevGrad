#include "Strategy.h"

namespace RevGrad {
    SGD::SGD(std::vector<Tensor> parameters, float learning_rate, float momentum) 
        : learning_rate(learning_rate), momentum(momentum)
    {
        this->parameters = parameters;
        int size = 0;
        for (auto param : parameters) {
            size += param.size();
        }
        this->velocity.resize(size);
    }

    void SGD::zero() {
        for (auto param : this->parameters) {
            for (auto& grad : param.grads()) {
                grad = 0.0f;
            }
        }
    }

    void SGD::update() {
        int j = 0;
        for (auto param : parameters) {
            for (int i = 0; i < param.size(); i++) {
                velocity[j] = momentum * velocity[j] - learning_rate * param.grads()[i];
                param.values()[i] += velocity[j];
                j++;
            }
        }
    }
}
