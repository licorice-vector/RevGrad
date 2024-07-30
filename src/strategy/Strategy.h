#ifndef STRATEGY_H
#define STRATEGY_H

#include "../tensor/Tensor.h"

namespace RevGrad {
    class Strategy {
    public:
        std::vector<Tensor> parameters;
        Strategy() {}
        virtual void zero() = 0;
        virtual void update() = 0;
    };

    class SGD : public Strategy {
        float learning_rate;
        float momentum;
        std::vector<float> velocity;
    public:
        SGD(std::vector<Tensor> parameters, float learning_rate, float momentum = 0.9);
        void zero() override;
        void update() override;
    };
}

#endif
