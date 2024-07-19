#ifndef MODEL_H
#define MODEL_H

#include <functional>

#include "../tensor/Tensor.h"

namespace RevGrad {
    class Model {
    public:
        std::vector<Float> parameters;
        Model() {}
        std::vector<Float> get_params();
        Tensor operator()(Tensor x);
        virtual Tensor forward(Tensor x) = 0;
    };

    class Linear : public Model {
    public:
        int in_features;
        int out_features;
        Tensor weights;
        Tensor bias;
        Linear() {}
        Linear(Model* parent_model, int in_features, int out_features);
        Tensor forward(Tensor x) override;
    };
}

#endif
