#include <stdio.h>

#include "NeuralNetwork.hpp"

int main(void) {
    NeuralNetwork nn;

    nn.addInputLayer(784);

    // nn.addHiddenLayer(28);
    // nn.addHiddenLayer(14);
    nn.addHiddenLayer(10);

    nn.addOutputLayer(10);

    Matrix input = Matrix(784, 4).rand(0.0, 255.0);
    Matrix output = Matrix(10, 4).zero();

    output[3][0] = 1.0;
    output[5][1] = 1.0;
    output[2][2] = 1.0;
    output[9][3] = 1.0;

    nn.gradientDescent(input, output, 500, 1e-2);

    return 0;
}