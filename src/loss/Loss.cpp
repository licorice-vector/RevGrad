#include "Loss.h"

namespace RevGrad {
    Tensor Loss::operator()(Tensor prediction, Tensor correct) {
        return compute(prediction, correct);
    }

    Tensor MSE::compute(Tensor prediction, Tensor correct) {
        int n = correct.size();
        assert(prediction.size() == n);

        prediction.flatten();
        correct.flatten();
        
        return Tensor::sum((prediction - correct) * (prediction - correct)) / (2.0f * n);
    }

    Tensor CrossEntropyLoss::compute(Tensor prediction, Tensor correct) {
        assert(prediction.shape() == correct.shape());
        return -Tensor::mean(Tensor::sum(correct * Tensor::log(prediction), 0));
    }

    Tensor NLLLoss::compute(Tensor prediction, Tensor correct) {
        assert(prediction.shape() == correct.shape());
        return -Tensor::mean(Tensor::sum(correct * prediction, 0));
    }
}
