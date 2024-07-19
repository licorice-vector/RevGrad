#ifndef STRATEGY_H
#define STRATEGY_H

#include "../float/Float.h"

namespace RevGrad {
    class Strategy {
    public:
        std::vector<Float> parameters;
        Strategy() {}
        virtual void zero() = 0;
        virtual void update() = 0;
    };

    class SGD : public Strategy {
        float learning_rate;
        float momentum;
        std::vector<float> velocity;
    public:
        SGD(std::vector<Float> parameters, float learning_rate, float momentum = 0.9);
        void zero() override;
        void update() override;
    };
}

#endif
