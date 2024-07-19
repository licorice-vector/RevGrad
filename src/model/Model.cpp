#include <iostream>
#include <cassert>

#include "Model.h"

namespace RevGrad {
    std::vector<Float> Model::get_params() {
        std::vector<Float> params;
        for (auto param : parameters) {
            params.push_back(param);
        }
        return params;
    }

    Tensor Model::operator()(Tensor x) {
        return forward(x);
    }

    Linear::Linear(Model* parent_model, int in_features, int out_features) 
        : in_features(in_features),
          out_features(out_features),
          weights(Tensor::random(Shape({out_features, in_features}), in_features)), 
          bias(Tensor::random(Shape({out_features, 1}), in_features))
    {
        for (auto param : weights.data) {
            parent_model->parameters.push_back(param);
        }
        for (auto param : bias.data) {
            parent_model->parameters.push_back(param);
        }
    }

    /*
        @param x tensor of shape (features, batch size)
    */
    Tensor Linear::forward(Tensor x) {
        return Tensor::matmul(weights, x) + bias;
    }
}
