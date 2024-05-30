#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

Matrix NeuralNetwork::expandBias(Matrix &bias, size_t cols) {
    Matrix res = Matrix(bias.getRows(), cols);

    for (size_t row = 0; row < bias.getRows(); row++)
        for (size_t col = 0; col < cols; col++)
            res[row][col] = bias[row][0];

    return res;
}

void NeuralNetwork::ReLU(Matrix &mat) {
    for (size_t row = 0; row < mat.getRows(); row++)
        for (size_t col = 0; col < mat.getCols(); col++)
            mat[row][col] = std::max(mat[row][col], 0.0);
}

void NeuralNetwork::softmax(Matrix &mat) {
    for (size_t row = 0; row < mat.getRows(); row++)
        for (size_t col = 0; col < mat.getCols(); col++)
            mat[row][col] = std::exp(mat[row][col]);

    double sum = mat.sum();

    for (size_t row = 0; row < mat.getRows(); row++)
        for (size_t col = 0; col < mat.getCols(); col++)
            mat[row][col] /= sum;
}

void NeuralNetwork::addInputLayer(size_t size) {
    if (SAFETY_CHECKS) {
        if (inputLayerSize > 0) {
            fprintf(stderr, "ERROR: Cannot add another input layer\n");
            exit(1);
        }

        if (size == 0) {
            fprintf(stderr, "ERROR: Input layer size cannot be zero\n");
            exit(1);
        }
    }

    inputLayerSize = size;
    numLayers++;
}

void NeuralNetwork::addHiddenLayer(size_t size) {
    if (SAFETY_CHECKS) {
        if (numLayers < 1) {
            fprintf(stderr, "ERROR: Cannot add hidden layer, add an input layer first\n");
            exit(1);
        }

        if (size == 0) {
            fprintf(stderr, "ERROR: Hidden layer size cannot be zero\n");
            exit(1);
        }
    }

    // Connect to input layer
    size_t lastLayerSize = inputLayerSize;

    // Connect to last hidden layer
    if (numLayers > 1)
        lastLayerSize = weights.back().getRows();
    
    weights.push_back(Matrix(size, lastLayerSize).rand());

    // Biases do not depend on last layer size
    biases.push_back(Matrix(size, 1).rand());
    
    numLayers++;
}

void NeuralNetwork::addOutputLayer(size_t size) {
    if (SAFETY_CHECKS) {
        if (outputLayerSize > 0) {
            fprintf(stderr, "ERROR: Cannot add another output layer\n");
            exit(1);
        }

        if (numLayers < 2) {
            fprintf(stderr, "ERROR: Cannot add output layer, add an input layer and at least one hidden layer first\n");
            exit(1);
        }

        if (size == 0) {
            fprintf(stderr, "ERROR: Output layer size cannot be zero\n");
            exit(1);
        }
    }

    // Connect to last hidden layer
    size_t lastLayerSize = weights.back().getRows();

    weights.push_back(Matrix(size, lastLayerSize).rand());

    // Biases do not depend on last layer size
    biases.push_back(Matrix(size, 1).rand());

    outputLayerSize = size;
    numLayers++;
}

Matrix NeuralNetwork::forwardPropagation(Matrix &input) {
    if (SAFETY_CHECKS) {
        if (numLayers < 3) {
            fprintf(stderr, "ERROR: Cannot propagate forward. A minimum of three layers (input, hidden+, output) is required\n");
            exit(1);
        }

        if (input.getRows() != inputLayerSize) {
            fprintf(stderr, "ERROR: Input data has incorrect dimensions. Expected: (%lld, Any) | Got: (%lld, %lld)\n",
            inputLayerSize,
            input.getRows(),
            input.getCols());
            exit(1);
        }
    }

    // Expand bias vector to every column
    Matrix B = expandBias(biases[0], input.getCols());

    // Input layer
    Matrix Z = weights[0].dot(input).add(B);
    
    // Hidden layers
    for (size_t layer = 1; layer < numLayers - 1; layer++) {
        Matrix B = expandBias(biases[layer], Z.getCols());

        ReLU(Z);
        Z = weights[layer].dot(Z).add(B); 
    }

    // Output layer
    softmax(Z);

    return Z;
}