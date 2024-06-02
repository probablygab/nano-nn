#include "NeuralNetwork.hpp"

/**
 * @brief Construct an empty Neural Network. 
 * Use addInputLayer, addHiddenLayer and addOutputLayer to build it.
 * 
 */
NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}


/**
 * @brief Add an input layer to the Neural Network.
 * 
 * @param size Size of the input layer. Should match the amount of features in the input data.
 */
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


/**
 * @brief Add a hidden layer to the Neural Network. Will connect to the previous hidden layer or input layer.
 * 
 * @param size Size of the hidden layer, can be any positive integer, usually between the input and output size.
 */
void NeuralNetwork::addHiddenLayer(size_t size) {
    if (SAFETY_CHECKS) {
        if (numLayers < 1) {
            fprintf(stderr, "ERROR: Cannot add hidden layer, add an input layer first\n");
            exit(1);
        }

        if (outputLayerSize > 0) {
            fprintf(stderr, "ERROR: Cannot add hidden layer, output layer already exists\n");
            exit(1);
        }

        if (size == 0) {
            fprintf(stderr, "ERROR: Hidden layer size cannot be zero\n");
            exit(1);
        }
    }

    // Connect to input layer by default
    size_t lastLayerSize = inputLayerSize;

    // Connect to last hidden layer if available, else will connect to input layer
    if (numLayers > 1)
        lastLayerSize = weights.back().getRows();
    
    weights.push_back(Matrix(size, lastLayerSize).rand());

    // Biases do not depend on last layer size
    biases.push_back(Matrix(size, 1).rand());
    
    numLayers++;
}


/**
 * @brief Add an output layer to the Neural Network. Will connect to the last hidden layer.
 * 
 * @param size Size of the output layer. Should match the amount of classes in the output data.
 */
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


/**
 * @brief Copy bias vector to every column of a Matrix.
 * This is used to apply the bias to every input in a batch.
 * 
 * @param bias Bias vector.
 * @param cols Amount of columns to copy the bias to.
 * @return A new Matrix with the bias vector copied to every column.
 */
Matrix NeuralNetwork::expandBias(const Matrix &bias, size_t cols) {
    Matrix res = Matrix(bias.getRows(), cols);

    #pragma omp parallel for
    for (size_t row = 0; row < bias.getRows(); row++)
        for (size_t col = 0; col < cols; col++)
            res[row][col] = bias[row][0];

    return res;
}


/**
 * @brief Rectified Linear Unit activation function.
 * 
 * @param mat Matrix to apply the ReLU function to (will be modified).
 * @return Reference to the same matrix.
 */
Matrix& NeuralNetwork::ReLU(Matrix &mat) {
    #pragma omp parallel for
    for (size_t row = 0; row < mat.getRows(); row++)
        for (size_t col = 0; col < mat.getCols(); col++)
            mat[row][col] = std::max(mat[row][col], 0.0);

    return mat;
}


/**
 * @brief Derivative of the Rectified Linear Unit activation function.
 * 
 * @param mat Matrix to apply the derivative of the ReLU function to (will be modified).
 * @return Reference to the same matrix.
 */
Matrix& NeuralNetwork::derivativeReLU(Matrix &mat) {
    #pragma omp parallel for
    for (size_t row = 0; row < mat.getRows(); row++) {
        for (size_t col = 0; col < mat.getCols(); col++) {
            if (mat[row][col] > 0)
                mat[row][col] = 1.0;
            else
                mat[row][col] = 0.0;
        }
    }

    return mat;
}


/**
 * @brief Softmax activation function.
 * Applies the softmax function to every column of the matrix.
 * Each column represents the output of a single input.
 * 
 * @param mat Matrix to apply the softmax function to (will be modified).
 * @return Reference to the same matrix.
 */
Matrix& NeuralNetwork::softmax(Matrix &mat) {
    // Apply softmax column by column
    #pragma omp parallel for
    for (size_t col = 0; col < mat.getCols(); col++) {
        // Offset by max to avoid inf and garbage results
        double max = mat[0][col];

        #pragma omp simd reduction(max:max)
        for (size_t row = 1; row < mat.getRows(); row++)
            max = std::max(mat[row][col], max);

        // Exp
        #pragma omp simd
        for (size_t row = 0; row < mat.getRows(); row++)
            mat[row][col] = std::exp(mat[row][col] - max);

        // Get sum and normalize
        double sum = 0.0;

        #pragma omp simd reduction(+:sum)
        for (size_t row = 0; row < mat.getRows(); row++)
            sum += mat[row][col];

        #pragma omp simd
        for (size_t row = 0; row < mat.getRows(); row++)
            mat[row][col] /= sum;
    }

    return mat;
}


/**
 * @brief Feed Forward the input data through the Neural Network.
 * 
 * @param input Input data, should have the same amount of rows as the input layer size.
 * @return struct ForwardData with the output and hidden layers values.
 */
ForwardData NeuralNetwork::forwardPropagation(const Matrix &input) {
    if (SAFETY_CHECKS) {
        if (numLayers < 3) {
            fprintf(stderr, "ERROR: Cannot propagate forward. A minimum of three layers (input, hidden+, output) is required\n");
            exit(1);
        }

        if (input.getRows() != inputLayerSize) {
            fprintf(stderr, "ERROR: Input data has incorrect dimensions. Expected: (%zu, Any) | Got: (%zu, %zu)\n",
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


/**
 * @brief Back Propagate the errors through the Neural Network, calculating delta weights and biases.
 * This is the main function to train the Neural Network, as it estimates the gradient of the loss function.
 * 
 * @param input Input data, should have the same amount of rows as the input layer size.
 * @param targetOutput Target output data, should have the same amount of rows as the output layer size.
 * @param forwardData ForwardData struct with the output and hidden layers values, from a previous forwardPropagation call.
 * @return struct BackData with delta weights and biases to update parameters.
 */
BackData NeuralNetwork::backPropagation(const Matrix &input, const Matrix& targetOutput, ForwardData &forwardData) {
    if (SAFETY_CHECKS) {
        if (numLayers < 3) {
            fprintf(stderr, "ERROR: Cannot propagate backwards. A minimum of three layers (input, hidden+, output) is required\n");
            exit(1);
        }

        if (input.getRows() != inputLayerSize) {
            fprintf(stderr, "ERROR: Input data has incorrect dimensions. Expected: (%zu, Any) | Got: (%zu, %zu)\n",
            inputLayerSize,
            input.getRows(),
            input.getCols());
            exit(1);
        }

        if (targetOutput.getRows() != outputLayerSize) {
            fprintf(stderr, "ERROR: output data has incorrect dimensions. Expected: (%zu, Any) | Got: (%zu, %zu)\n",
            outputLayerSize,
            targetOutput.getRows(),
            targetOutput.getCols());
            exit(1);
        }

        if (input.getCols() != targetOutput.getCols()) {
            fprintf(stderr, "ERROR: input and output data must have the same amount of columns. Current: (%zu, %zu *) vs. (%zu, %zu *)\n",
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
    
    // Amount of entries being processed at once (one entry per column)
    double m = input.getCols();

    // Output layer (we copy the output matrix since sub modifies it)
    Matrix dZ = Matrix(forwardData.output).sub(targetOutput); 

    // Hidden layers
    for (size_t layer = numLayers - 2; layer > 0; layer--) {
        // Calculate deltas
        result.deltaWeights[layer] = dZ.dotTransposeRight(forwardData.hiddenValuesAfterActivation[layer - 1]).div(m);
        result.deltaBiases[layer] = dZ.sum() / m;

        // Advance to previous layer (copy weight Matrix to avoid modifying it)
        dZ = Matrix(weights[layer]).transpose().dot(dZ).mult(derivativeReLU(forwardData.hiddenValues[layer - 1]));
    }

    // Input layer
    result.deltaWeights[0] = dZ.dotTransposeRight(input).div(m);
    result.deltaBiases[0] = dZ.sum() / m;

    return result;
}


/**
 * @brief Calculate the accuracy of the Neural Network output compared to the target output.
 * The output is considered correct if the highest value in the output matrix corresponds to a 1.0 in the target output matrix.
 * 
 * @param output Output matrix, from a previous forwardPropagation call.
 * @param targetOutput Target output data.
 * @return Accuracy as a percentage [0.0, 1.0].
 */
double NeuralNetwork::calculateAccuracy(const Matrix& output, const Matrix& targetOutput) const {
    if (SAFETY_CHECKS)
        if (output.getRows() != targetOutput.getRows() || output.getCols() != targetOutput.getCols()) {
            fprintf(stderr, "ERROR: Cannot calculate accuracy for output matrices of different dimensions: (%zu, %zu) vs. (%zu, %zu)\n",
            output.getRows(), output.getCols(), targetOutput.getRows(), targetOutput.getCols());
            exit(1);
        }

    // Amount of correct output values
    double targetCorrect = targetOutput.sum();

    // Amount of wrong output values 
    double targetWrong = targetCorrect;

    // Column by column, find max element from output
    for (size_t col = 0; col < output.getCols(); col++) {
        size_t maxIdx = 0;

        for (size_t row = 1; row < output.getRows(); row++)
            if (output[row][col] > output[maxIdx][col])
                maxIdx = row;

        // If at this location, targetOutput is 1.0, then output is correct
        if (targetOutput[maxIdx][col] > 0.9)
            targetWrong -= 1.0;
    }

    // Calculate accuracy 
    return (targetCorrect - targetWrong) / targetCorrect;
}


/**
 * @brief Get the index of the highest value in the output matrix.
 * This is used to predict the class of the input data.
 * 
 * @param output Output matrix, from a previous forwardPropagation call.
 * @return Index of the highest value in the output matrix.
 */
size_t NeuralNetwork::getPrediction(const Matrix& output) const {
    size_t maxIdx = 0;

    for (size_t row = 1; row < output.getRows(); row++)
        if (output[row][0] > output[maxIdx][0])
            maxIdx = row;

    return maxIdx;
}


/**
 * @brief Update the Neural Network parameters (weights and biases) using the calculated deltas.
 * 
 * @param backData BackData struct with delta weights and biases, from a previous backPropagation call.
 * @param learningRate Learning rate to apply to the deltas.
 */
void NeuralNetwork::updateParameters(BackData &backData, double learningRate) {
    for (size_t layer = 0; layer < numLayers - 1; layer++) {
        weights[layer].sub(backData.deltaWeights[layer].mult(learningRate));
        biases[layer].sub(backData.deltaBiases[layer] * learningRate);
    }
}


/**
 * @brief Train the Neural Network using the input data and target output data.
 * A gradient descent is applied to minimize the loss function.
 * @n
 * The amount of columns in the input matrix represents the amount of entries to process at once.
 * Ideally, this should be the whole dataset, but it can be a batch of entries.
 * @n
 * If any column from the output matrix is full of zeros, the Neural Network will train erratically.
 * Make sure every input has a class in the output matrix.
 * 
 * @param input Input data, should have the same amount of rows as the input layer size.
 * @param targetOutput Target output data, should have the same amount of rows as the output layer size.
 * @param iterations Amount of iterations to perform.
 * @param learningRate Learning rate to apply to the deltas.
 */
void NeuralNetwork::gradientDescent(const Matrix &input, const Matrix& targetOutput, size_t iterations, double learningRate) {
    for (size_t iter = 0; iter < iterations; iter++) {
        ForwardData fwdData = forwardPropagation(input);
        BackData backData = backPropagation(input, targetOutput, fwdData);

        // Update gradually
        updateParameters(backData, learningRate);

        if (iter % ITER_FEEDBACK_FREQUENCY == 0 || iter == iterations - 1) {
            // Accuracy
            double accuracy = calculateAccuracy(fwdData.output, targetOutput);

            // Mean Squared Error (this will change the output Matrix in-place)
            fwdData.output.sub(targetOutput);
            double loss = fwdData.output.mult(fwdData.output).sum() / fwdData.output.getSize();

            printf("Iteration: %zu  Loss (MSE): %lf  Accuracy: %.2lf%%\n", iter, loss, 100.0 * accuracy);
        }
    }
}


/**
 * @brief Save the Neural Network parameters (weights and biases) to a binary file.
 * @attention The Neural Network configuration IS NOT saved, only the parameters.
 * 
 * @param filepath File path to save parameters.
 */
void NeuralNetwork::saveParameters(const char* filepath) {
    FILE* fp = fopen(filepath, "wb");

    if (fp == NULL) {
        fprintf(stderr, "WARNING: Could not create parameters file: %s\n", filepath);
        return;
    }

    // Dump arrays in pure binary fashion
    for (const Matrix &wei : weights) {
        const void* data = &wei[0][0];

        fwrite(data, wei.getSize() * sizeof(double), 1, fp);
    }

    for (const Matrix &bias : biases) {
        const void* data = &bias[0][0];

        fwrite(data, bias.getSize() * sizeof(double), 1, fp);
    }

    fclose(fp);    
}


/**
 * @brief Load the Neural Network parameters (weights and biases) from a binary file.
 * @attention The Neural Network configuration IS NOT loaded, only the parameters.
 * Absolute NO CHECKS are made to ensure the parameters are compatible with the Neural Network.
 * 
 * You need to initialize the Neural Network with the same input size, hidden layers and output size of the file you are loading.
 * 
 * @param filepath File path to load parameters from.
 */
void NeuralNetwork::loadParameters(const char* filepath) {
    FILE* fp = fopen(filepath, "rb");

    if (fp == NULL) {
        fprintf(stderr, "WARNING: Could not open parameters file: %s\n", filepath);
        return;
    }

    // Load arrays in pure binary fashion
    for (Matrix &wei : weights) {
        void* data = &wei[0][0];

        fread(data, wei.getSize() * sizeof(double), 1, fp);
    }

    for (Matrix &bias : biases) {
        void* data = &bias[0][0];

        fread(data, bias.getSize() * sizeof(double), 1, fp);
    }

    fclose(fp);    
}


/**
 * @brief Get a view of the weights from the Neural Network.
 * 
 * @return const reference to the weights.
 */
const std::vector<Matrix>& NeuralNetwork::getWeightsView() const {
    return weights;
}


/**
 * @brief Get a view of the biases from the Neural Network.
 * 
 * @return const reference to the biases.
 */
const std::vector<Matrix>& NeuralNetwork::getBiasesView() const {
    return biases;
}