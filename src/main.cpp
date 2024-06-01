// Nano Multi-Layer Perceptron Neural Network by Gab - June 2024 (absolutely not optimized)
#include "NeuralNetwork.hpp"
#include "demoView.hpp"


/**
 * @brief Parameters for the Neural Network.
 * Tailored to the MNIST dataset.
 * 
 */
#define INPUT_SIZE              784
#define OUTPUT_SIZE             10

#define TOTAL_TRAIN_ENTRIES     31000
#define TOTAL_TEST_ENTRIES      11000

#define TRAIN_CSV_PATH          "../data/train.csv"
#define TEST_CSV_PATH           "../data/test.csv"
#define PARAMETERS_FILE_PATH    "nn-parameters-784-56-28-10.bin"


/**
 * @brief Input data struct. 
 * Holds the input and output matrices for the Neural Network.
 * 
 * @attention
 * This struct relies on a custom function to read CSV files.
 * You need to implement your own function if you want to use this struct with different data.
 * 
 */
typedef struct InputData {
    Matrix input;
    Matrix output;
} InputData;


/**
 * @brief Read a CSV file and return its data as input and output matrices.
 * 
 * @attention This particular function is tailored to read the MNIST dataset, which has the following format:
 * @n
 * First column is the label (0-9).
 * Remaining 784 columns are pixel values (0-255) for 28x28 images.
 * 
 * @param filepath Path to the CSV file.
 * @param inputSize Size of the input vector (In this case: 784 pixels).
 * @param outputSize Size of the output vector (In this case: 10 classes).
 * @param totalEntries Total number of entries in the CSV file, can be smaller if you wish to read less data.
 * @return Input data struct with formatted and normalized input and output matrices.
 */
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
    
    // Read col lines (one entry per CSV line -> one entry per input column)
    for (size_t col = 0; col < totalEntries; col++) {
        int value;

        // First column is "label"
        int read = fscanf(fp, "%d,", &value);

        // Reached EOF. This prevents reading beyond the last entry
        // if totalEntries is larger than the file.
        if (read == 0)
            break;

        // Label as the index of output vector
        data.output[value][col] = 1.0;

        for (size_t row = 0; row < inputSize; row++) {
            // Remaining n values are input data
            fscanf(fp, "%d,", &value);

            data.input[row][col] = (double) value / 255.0; // Max pixel values is 255
        }
    }

    printf("Done!\n");

    fclose(fp);

    return data;
}


int main(int argc, char* argv[]) {
    // Init Neural Network
    NeuralNetwork nn;

    nn.addInputLayer(INPUT_SIZE);

    // Arbitrary sizes, you can change this
    nn.addHiddenLayer(56);
    nn.addHiddenLayer(28);

    nn.addOutputLayer(OUTPUT_SIZE);

    // Run visual demo by default if no arguments are passed
    if (argc < 2) {
        // Demo parameters (90% accuracy on test data)
        nn.loadParameters(PARAMETERS_FILE_PATH);

        runDemo(nn);
        return 0;
    }

    // If any argument is passed, train the Neural Network
    
    // Read MNIST dataset
    InputData train = readCSV(TRAIN_CSV_PATH, INPUT_SIZE, OUTPUT_SIZE, TOTAL_TRAIN_ENTRIES);
    InputData test = readCSV(TEST_CSV_PATH, INPUT_SIZE, OUTPUT_SIZE, TOTAL_TEST_ENTRIES);

    // Train
    nn.gradientDescent(train.input, train.output, 600, 1e-1);

    // Test NN accuracy on test data
    ForwardData result = nn.forwardPropagation(test.input);

    double acc = nn.calculateAccuracy(result.output, test.output);

    printf("Accuracy on test data: %.2lf%%\n", 100.0 * acc);

    nn.saveParameters(PARAMETERS_FILE_PATH);

    return 0;
}