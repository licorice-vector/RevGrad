#include <cassert>
#include <iostream>

#include "Loss.h"

namespace RevGrad {
    Tensor Loss::operator()(Tensor prediction, Tensor correct) {
        return compute(prediction, correct);
    }

    Tensor MSE::compute(Tensor prediction, Tensor correct) {
        assert(prediction.shape.size() == correct.shape.size());
        prediction = prediction.flattened();
        correct = correct.flattened();
        int n = correct.shape.size();
        Float mse;
        for (int i = 0; i < n; i++) {
            Float x = prediction[{i}] - correct[{i}];
            mse = mse + x * x;
        }
        mse = mse / (2.0f * n); // division by 2 makes derivative more stable
        return Tensor(mse);
    }
}
