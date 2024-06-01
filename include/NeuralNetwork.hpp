#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix.hpp"

#include <vector>
#include <math.h>

#define ITER_FEEDBACK 10

typedef struct ForwardData {
    Matrix output;
    std::vector<Matrix> hiddenValues;
    std::vector<Matrix> hiddenValuesAfterActivation;
} ForwardData;

typedef struct BackData {
    std::vector<Matrix> deltaWeights;
    std::vector<double> deltaBiases;
} BackData;

class NeuralNetwork {
    public:
        size_t numLayers = 0;
        size_t inputLayerSize = 0;
        size_t outputLayerSize = 0;

        std::vector<Matrix> weights;
        std::vector<Matrix> biases;  

    private:
        Matrix expandBias(Matrix &bias, size_t cols);

        Matrix& ReLU(Matrix &mat);
        Matrix& derivativeReLU(Matrix &mat);

        Matrix& softmax(Matrix &mat);     

    public:
        NeuralNetwork();
        ~NeuralNetwork();

        void addInputLayer(size_t size);
        void addHiddenLayer(size_t size);
        void addOutputLayer(size_t size);

        ForwardData forwardPropagation(const Matrix &input);
        BackData backPropagation(const Matrix &input, const Matrix& targetOutput, ForwardData &forwardData);

        double calculateAccuracy(const Matrix& output, const Matrix& targetOutput) const;
        size_t getPrediction(const Matrix& output) const;

        void updateParameters(BackData &backData, double learningRate);
        void gradientDescent(const Matrix &input, const Matrix& targetOutput, size_t iterations, double learningRate);

        void saveParameters(const char* filepath);
        void loadParameters(const char* filepath);
};

#endif