#include <iostream>
#include <cassert>

#include "Strategy.h"

namespace RevGrad {
    SGD::SGD(std::vector<Float> parameters, float learning_rate, float momentum) 
        : learning_rate(learning_rate), momentum(momentum)
    {
        this->parameters = parameters;
        this->velocity.resize(parameters.size());
    }

    void SGD::zero() {
        for (auto param : this->parameters) {
            param.set_grad(0);
        }
    }

    void SGD::update() {
        for (int i = 0; i < parameters.size(); i++) {
            velocity[i] = momentum * velocity[i] + learning_rate * parameters[i].grad();
            parameters[i].set_value(parameters[i].value() - velocity[i]);
        }
    }
}
