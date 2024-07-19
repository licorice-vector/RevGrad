#ifndef LOSS_H
#define LOSS_H

#include "../tensor/Tensor.h"

namespace RevGrad {
    class Loss {
    public:
        Loss() {}
        Tensor operator()(Tensor prediction, Tensor correct);
        virtual Tensor compute(Tensor prediction, Tensor correct) = 0;
    };

    class MSE : public Loss {
    public:
        MSE() {}
        Tensor compute(Tensor prediction, Tensor correct) override;
    };
}

#endif
