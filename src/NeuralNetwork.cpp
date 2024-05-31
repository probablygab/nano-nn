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

Matrix& NeuralNetwork::ReLU(Matrix &mat) {
    for (size_t row = 0; row < mat.getRows(); row++)
        for (size_t col = 0; col < mat.getCols(); col++)
            mat[row][col] = std::max(mat[row][col], 0.0);

    return mat;
}

Matrix& NeuralNetwork::derivativeReLU(Matrix &mat) {
    for (size_t row = 0; row < mat.getRows(); row++)
        for (size_t col = 0; col < mat.getCols(); col++) {
            if (mat[row][col] > 0)
                mat[row][col] = 1.0;
            else
                mat[row][col] = 0.0;
        }

    return mat;
}

Matrix& NeuralNetwork::softmax(Matrix &mat) {
    // Apply softmax column by column
    for (size_t col = 0; col < mat.getCols(); col++) {
        // Offset by max to avoid inf and garbage results
        double max = mat[0][col];

        for (size_t row = 1; row < mat.getRows(); row++)
            if (mat[row][col] > max)
                max = mat[row][col];

        // Exp
        for (size_t row = 0; row < mat.getRows(); row++)
            mat[row][col] = std::exp(mat[row][col] - max);

        // Get sum and normalize
        double sum = 0.0;

        for (size_t row = 0; row < mat.getRows(); row++)
            sum += mat[row][col];

        for (size_t row = 0; row < mat.getRows(); row++)
            mat[row][col] /= sum;
    }


    // // Offset by max to avoid inf and garbage results
    // double max = mat.max();

    // for (size_t row = 0; row < mat.getRows(); row++)
    //     for (size_t col = 0; col < mat.getCols(); col++)
    //         mat[row][col] = std::exp(mat[row][col] - max);

    // // Get sum and normalize
    // double sum = mat.sum();

    // for (size_t row = 0; row < mat.getRows(); row++)
    //     for (size_t col = 0; col < mat.getCols(); col++)
    //         mat[row][col] /= sum;

    return mat;
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

ForwardData NeuralNetwork::forwardPropagation(const Matrix &input) {
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

    // Result: output and intermediate layers values
    ForwardData result;

    // Expand bias vector to every column
    Matrix B = expandBias(biases[0], input.getCols());

    // Input layer
    Matrix Z = weights[0].dot(input).add(B);
    
    // Hidden layers
    for (size_t layer = 1; layer < numLayers - 1; layer++) {
        Matrix B = expandBias(biases[layer], Z.getCols());

        // Save values for visualization and back propagation
        result.hiddenValues.push_back(Z);
        ReLU(Z);
        result.hiddenValuesAfterActivation.push_back(Z);

        // Advance to next layer
        Z = weights[layer].dot(Z).add(B); 
    }

    // Output layer
    softmax(Z);
    result.output = Z;

    return result;
}

BackData NeuralNetwork::backPropagation(const Matrix &input, const Matrix& targetOutput, ForwardData &forwardData) {
    if (SAFETY_CHECKS) {
        if (numLayers < 3) {
            fprintf(stderr, "ERROR: Cannot propagate backwards. A minimum of three layers (input, hidden+, output) is required\n");
            exit(1);
        }

        if (input.getRows() != inputLayerSize) {
            fprintf(stderr, "ERROR: Input data has incorrect dimensions. Expected: (%lld, Any) | Got: (%lld, %lld)\n",
            inputLayerSize,
            input.getRows(),
            input.getCols());
            exit(1);
        }

        if (targetOutput.getRows() != outputLayerSize) {
            fprintf(stderr, "ERROR: output data has incorrect dimensions. Expected: (%lld, Any) | Got: (%lld, %lld)\n",
            outputLayerSize,
            targetOutput.getRows(),
            targetOutput.getCols());
            exit(1);
        }

        if (input.getCols() != targetOutput.getCols()) {
            fprintf(stderr, "ERROR: input and output data must have the same amount of columns. Current: (%lld, %lld *) vs. (%lld, %lld *)\n",
            input.getRows(),
            input.getCols(),
            targetOutput.getRows(),
            targetOutput.getCols());
            exit(1);
        }
    }

    // Result: delta weights and biases to update parameters
    BackData result;
    result.deltaWeights.resize(numLayers - 1);
    result.deltaBiases.resize(numLayers - 1);
    
    // Amount of entries being processed at once
    double m = input.getCols();

    // Output layer
    Matrix dZ = forwardData.output.sub(targetOutput); 

    // Hidden layers
    for (size_t layer = numLayers - 2; layer > 0; layer--) {
        // Calculate deltas
        result.deltaWeights[layer] = dZ.dot(forwardData.hiddenValuesAfterActivation[layer - 1].transpose()).div(m);
        result.deltaBiases[layer] = dZ.sum() / m;

        // Advance to previous layer
        dZ = weights[layer].transpose().dot(dZ).mult(derivativeReLU(forwardData.hiddenValues[layer - 1]));
    }

    // Input layer
    result.deltaWeights[0] = dZ.dot(input.transpose()).div(m);
    result.deltaBiases[0] = dZ.sum() / m;

    return result;
}

void NeuralNetwork::updateParameters(BackData &backData, double learningRate) {
    for (size_t layer = 0; layer < numLayers - 1; layer++) {
        weights[layer].sub(backData.deltaWeights[layer].mult(learningRate));
        biases[layer].sub(backData.deltaBiases[layer] * learningRate);
    }
}

void NeuralNetwork::gradientDescent(const Matrix &input, const Matrix& targetOutput, size_t iterations, double learningRate) {
    for (size_t iter = 0; iter < iterations; iter++) {
        ForwardData fwdData = forwardPropagation(input);
        BackData backData = backPropagation(input, targetOutput, fwdData);

        // Update gradually
        updateParameters(backData, learningRate);

        if (iter % ITER_FEEDBACK == 0) {
            // Mean Squared Error
            fwdData.output.sub(targetOutput);

            double loss = fwdData.output.mult(fwdData.output).sum() / fwdData.output.getSize();

            printf("Iteration: %lld  Loss (MSE): %.2lf\n", iter, loss);
        }
    }
}