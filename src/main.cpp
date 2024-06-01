// Nano Multi-Layer Perceptron Neural Network by Gab - June 2024 (absolutely not optimized)
#include "NeuralNetwork.hpp"
#include "demoView.hpp"

typedef struct InputData {
    Matrix input;
    Matrix output;
} InputData;

InputData readCSV(const char* filepath, size_t inputSize, size_t outputSize, size_t totalEntries) {
    InputData data = {Matrix(inputSize, totalEntries), Matrix(outputSize, totalEntries).zero()};

    FILE* fp = fopen(filepath, "r");

    if (fp == NULL) {
        fprintf(stderr, "ERROR: Could not open file: %s\n", filepath);
        exit(1);
    }

    printf("Reading %s ... ", filepath);

    // Skip first line
    while (fgetc(fp) != '\n')
        ;
    
    // Read col lines (one entry per line, one entry per column)
    for (size_t col = 0; col < totalEntries; col++) {
        int value;

        // First column is "label"
        int read = fscanf(fp, "%d,", &value);

        // EOF
        if (read == 0)
            break;

        // Label as the index of output vector
        data.output[value][col] = 1.0;

        for (size_t row = 0; row < inputSize; row++) {
            // Remaining n values are input data
            fscanf(fp, "%d,", &value);

            data.input[row][col] = (double) value / 255.0; 
        }
    }

    printf("Done!\n");

    fclose(fp);

    return data;
}

int main(void) {
    // Init Neural Network
    NeuralNetwork nn;

    nn.addInputLayer(784);

    nn.addHiddenLayer(56);
    nn.addHiddenLayer(28);

    nn.addOutputLayer(10);

    // runDemo(nn);

    InputData train = readCSV("../data/train.csv", 784, 10, 310); // 31000 total entries
    InputData test = readCSV("../data/test.csv", 784, 10, 110);   // 11000 total entries

    // Train
    nn.gradientDescent(train.input, train.output, 600, 1e-1);

    // Test NN accuracy on test data
    ForwardData result = nn.forwardPropagation(test.input);

    double acc = nn.calculateAccuracy(result.output, test.output);

    printf("Accuracy on test data: %.2lf%%\n", 100.0 * acc);

    // nn.saveParameters("nn-parameters-784-56-28-10.bin");

    return 0;
}