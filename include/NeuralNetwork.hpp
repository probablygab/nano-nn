#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix.hpp"

#include <vector>
#include <math.h>

class NeuralNetwork {
    private:
        size_t numLayers = 0;
        size_t inputLayerSize = 0;
        size_t outputLayerSize = 0;

        std::vector<Matrix> weights;
        std::vector<Matrix> biases;  

    private:
        Matrix expandBias(Matrix &bias, size_t cols);

        void ReLU(Matrix &mat);
        //void derivativeReLU(Matrix &mat)

        void softmax(Matrix &mat);     

    public:
        NeuralNetwork();
        ~NeuralNetwork();

        void addInputLayer(size_t size);
        void addHiddenLayer(size_t size);
        void addOutputLayer(size_t size);

        Matrix forwardPropagation(Matrix &input);
};

#endif