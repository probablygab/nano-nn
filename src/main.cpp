#include <stdio.h>

#include "NeuralNetwork.hpp"

int main(void) {
    NeuralNetwork nn;

    nn.addInputLayer(784);

    nn.addHiddenLayer(28);

    nn.addOutputLayer(10);

    Matrix input = Matrix(784, 10).rand();

    Matrix output = nn.forwardPropagation(input);

    output.print();

    return 0;
}