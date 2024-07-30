#include <iostream>

#include "../utill/Print.h"
#include "../tensor/Tensor.h"
#include "../model/Model.h"
#include "../loss/Loss.h"
#include "../strategy/Strategy.h"

using namespace RevGrad;

class NN : public Model {
public:
    Linear l1;
    Linear l2;
    Linear l3;
    
    NN() {
        l1 = Linear(this, 2, 8);
        l2 = Linear(this, 8, 4);
        l3 = Linear(this, 4, 1);
    }

    Tensor forward(Tensor x) {
        Tensor y = x;
        y = l1(y);
        y = Tensor::relu(y);
        y = l2(y);
        y = Tensor::relu(y);
        y = l3(y);
        y = Tensor::sigmoid(y);
        return y;
    }
};

int main() {

    Tensor X = Tensor(Shape({8, 2}), {
        1, 0,
        1, 0, 
        1, 0, 
        1, 0, 
        0, 1, 
        0, 1, 
        0, 1, 
        0, 1
    });
    X.transpose();

    Tensor correct = Tensor(Shape({8}), {1, 1, 1, 1, 0, 0, 0, 0});

    NN nn;
    MSE mse;
    SGD sgd(nn.get_params(), 0.1);

    for (int i = 0; i <= 500; i++) {
        Tensor prediction = nn(X);
        prediction.flatten();
        Tensor loss = mse(prediction, correct);

        sgd.zero();
        loss.backward();
        sgd.update();
        
        if (i % 100 == 0) {
            std::cout << "prediction: " << prediction << std::endl;
            std::cout << "correct: " << correct << std::endl;
            std::cout << "loss: " << loss.value({0}) << std::endl;
        }
    }
    
    return 0;
}
