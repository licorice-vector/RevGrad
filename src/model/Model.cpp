#include "Model.h"

namespace RevGrad {
    std::vector<Tensor> Model::get_params() {
        std::vector<Tensor> params;
        for (auto& param : parameters) {
            params.push_back(param);
        }
        return params;
    }

    Tensor Model::operator()(Tensor x) {
        return forward(x);
    }

    void Model::save_parameters(const std::string& filename) {
        // TODO
    }

    void Model::load_parameters(const std::string& filename) {
        // TODO
    }

    Linear::Linear(Model* parent_model, int in_features, int out_features) 
        : in_features(in_features),
          out_features(out_features),
          weights(Tensor::random(Shape({out_features, in_features}), in_features)), 
          bias(Tensor(Shape({out_features, 1})))
    {
        parent_model->parameters.push_back(weights);
        parent_model->parameters.push_back(bias);
    }

    Tensor Linear::forward(Tensor x) {
        return Tensor::matmul(weights, x) + bias;
    }
}
