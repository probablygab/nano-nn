#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix.hpp"

#include <vector>
#include <math.h>


// Number of iterations between every feedback
#define ITER_FEEDBACK_FREQUENCY     10


/**
 * @brief Forward Data struct.
 * Holds the output matrix and hidden values for every layer.
 * @n
 * Returned by the forwardPropagation method. Used by the backPropagation method.
 * 
 */
typedef struct ForwardData {
    Matrix output;
    std::vector<Matrix> hiddenValues;
    std::vector<Matrix> hiddenValuesAfterActivation;
} ForwardData;


/**
 * @brief Back Data struct.
 * Holds the delta weights and biases for every layer.
 * @n
 * Returned by the backPropagation method. Used by the gradientDescent method.
 * 
 */
typedef struct BackData {
    std::vector<Matrix> deltaWeights;
    std::vector<double> deltaBiases;
} BackData;


/**
 * @brief Neural Network class. 
 * This Neural Network implementation is fully connected and uses ReLU activation for hidden layers and softmax for the output layer.
 * A few optimizations were made, but nothing major. This class is not faster than TensorFlow or PyTorch.
 * @attention
 * You need to build the network layer by layer, starting with the input layer.
 * Then you can feed data and train the network.
 * 
 */
class NeuralNetwork {
    private:
        size_t numLayers = 0;
        size_t inputLayerSize = 0;
        size_t outputLayerSize = 0;

        std::vector<Matrix> weights;
        std::vector<Matrix> biases;  

    private:
        Matrix expandBias(const Matrix &bias, size_t cols);

        Matrix& ReLU(Matrix &mat);
        Matrix& derivativeReLU(Matrix &mat);

        Matrix& softmax(Matrix &mat);    

        void updateParameters(BackData &backData, double learningRate); 

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

        void gradientDescent(const Matrix &input, const Matrix& targetOutput, size_t iterations, double learningRate);

        void saveParameters(const char* filepath);
        void loadParameters(const char* filepath);

        const std::vector<Matrix>& getWeightsView() const;
        const std::vector<Matrix>& getBiasesView() const;
};

#endif