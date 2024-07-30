#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <unistd.h>

#include "../utill/Print.h"
#include "../tensor/Tensor.h"
#include "../model/Model.h"
#include "../loss/Loss.h"
#include "../strategy/Strategy.h"

using namespace RevGrad;

class FeedForward : public Model {
public:
    Linear l1;
    Linear l2;
    Linear l3;
    
    FeedForward() {
        l1 = Linear(this, 784, 128);
        l2 = Linear(this, 128, 64);
        l3 = Linear(this, 64, 10);
    }

    Tensor forward(Tensor x) {
        Tensor y = x;
        y = l1(y);
        y = Tensor::relu(y);
        y = l2(y);
        y = Tensor::relu(y);
        y = l3(y);
        y = Tensor::log_softmax(y);
        return y;
    }
};

void print_image(Tensor image) {
    assert(image.size() == 784);
    image.reshape({28, 28});
    const std::string intensity_chars = " .:-=+*#%@";
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            float pixel_value = image.value({i, j});
            int char_index = static_cast<int>(pixel_value * (intensity_chars.size() - 1));
            std::cout << intensity_chars[char_index] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {

    // Data loading and processing
    Tensor train = Tensor::from_csv("examples/data/mnist_train.csv");
    Tensor test = Tensor::from_csv("examples/data/mnist_test.csv");

    auto split = [] (const Tensor& data) -> std::pair<Tensor, Tensor> {
        Tensor X = data.slice({{0, data.shape()[0]}, {1, data.shape()[1]}});
        float mx = 0;
        for (int i = 0; i < X.shape()[0]; i++) {
            for (int j = 0; j < X.shape()[1]; j++) {
                X.value({i, j}) /= 255.0f;
            }
        }
        Tensor y = data.slice({{0, data.shape()[0]}, {0, 1}});
        std::vector<float> values;
        for (int i = 0; i < y.shape()[0]; i++) {
            for (int j = 0; j < 10; j++) {
                values.push_back(y.value({i, 0}) == j);
            }
        }
        return {X, Tensor(Shape({y.shape()[0], 10}), values)};
    };

    auto [X_train, y_train] = split(train);
    auto [X_test, y_test] = split(test);

    std::cout << "X_train.shape: " << X_train.shape() << std::endl;
    std::cout << "y_train.shape: " << y_train.shape() << std::endl;
    std::cout << "X_test.shape: " << X_test.shape() << std::endl;
    std::cout << "y_test.shape: " << y_test.shape() << std::endl;

    // Model and number of model parameters
    FeedForward model;

    int parameter_cnt = 0;
    for (auto param : model.get_params()) {
        parameter_cnt += param.size();
    }
    std::cout << "Number of model parameters: " << parameter_cnt << std::endl;

    // Training
    NLLLoss nll_loss;
    SGD sgd(model.get_params(), 0.002f);

    int num_epochs = 4;
    int batch_size = 64;

    for (int i = 0; i < num_epochs; i++) {

        float epoch_loss = 0.0f;
        
        for (int j = 0; j < X_train.shape()[0]; j += batch_size) {

            Tensor batch = X_train.slice({{j, std::min(X_train.shape()[0], j + batch_size)}, {0, 784}});
            batch.transpose();
            Tensor correct = y_train.slice({{j, std::min(y_train.shape()[0], j + batch_size)}, {0, 10}});
            correct.transpose();
            
            Tensor prediction = model(batch);
            Tensor loss = nll_loss(prediction, correct);
            
            sgd.zero();
            loss.backward();
            sgd.update();
            
            epoch_loss += loss.value({0});
        }

        epoch_loss /= X_train.shape()[0];
        std::cout << "Epoch: " << i + 1 << ", training loss: " << epoch_loss << std::endl;
    }

    // Test accuracy
    X_test.transpose();
    Tensor prediction = model(X_test);
    prediction.transpose();

    auto get_prediction = [&] (int i) -> float {
        float best = std::numeric_limits<float>::lowest();
        float x = 0;
        for (int j = 0; j < 10; j++) {
            float current = prediction.value({i, j});
            if (current > best) {
                best = current;
                x = j;
            }
        }
        return x;
    };
    
    float accuracy = 0.0f;
    for (int i = 0; i < y_test.shape()[0]; i++) {
        if (test.value({i, 0}) == get_prediction(i)) {
            accuracy++;
        }
    }
    accuracy /= y_test.shape()[0];

    std::cout << "Test accuracy: " << (accuracy * 100.0f) << "%" << std::endl;

    // Test predictions visualized
    int n = 5;
    std::cout << n << " test predictions and correct" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "prediction: " << get_prediction(i) << std::endl;
        std::cout << "correct: " << test.value({i, 0}) << std::endl;
        std::cout << "image: " << std::endl;
        Tensor image = X_test.slice({{0, 784}, {i, i + 1}});
        print_image(image);
    }
    
    return 0;
}
