#include <stdio.h>

// Need a namespace for raylib because it also uses a struct called Matrix
// This way we can avoid conflicts
namespace ray {
#include "raylib/raylib.h"
}

#include "NeuralNetwork.hpp"

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

void drawWeights(const NeuralNetwork &nn, int x, int y) {
    for (const Matrix& weight : nn.weights) {
        std::string description = "Weight matrix: " + std::to_string(weight.getRows()) + "x" + std::to_string(weight.getCols());

        ray::DrawText(description.c_str(), x, y, 15, ray::WHITE);
        y += 20;

        for (size_t row = 0; row < weight.getRows(); row++) {
            for (size_t col = 0; col < weight.getCols(); col++) {
                double value = weight[row][col];

                ray::DrawPixel(x + col, y + row, ray::ColorAlpha(value > 0.0 ? ray::GREEN : ray::RED, (float) fabs(value)));
            }
        }

        y += weight.getRows() + 50;
    }
}

void drawNeurons(const ForwardData &result, int x, int y) {
    const int yOld = y;
    const int radius = 10;

    for (size_t i = 0; i < result.hiddenValuesAfterActivation.size(); i++) {
        std::string description = "Hidden layer " + std::to_string(i + 1) + " values: " + std::to_string(result.hiddenValuesAfterActivation[i].getRows());

        ray::DrawText(description.c_str(), x, y, 15, ray::WHITE);
        y += 30;

        for (size_t row = 0; row < result.hiddenValuesAfterActivation[i].getRows(); row++) {
            double value = result.hiddenValuesAfterActivation[i][row][0];

            ray::DrawCircle(x, y + row, radius, ray::ColorAlpha(ray::BLUE, (float) fabs(value)));
            ray::DrawCircleLines(x, y + row, radius, ray::BLUE);

            y += radius * 3;
        }

        y = yOld;
        x += 200;
    }

    std::string description = "Output layer values: " + std::to_string(result.output.getRows());

    ray::DrawText(description.c_str(), x, y, 15, ray::WHITE);
    y += 30;

    for (size_t row = 0; row < result.output.getRows(); row++) {
        double value = result.output[row][0];

        ray::DrawCircle(x, y + row, radius, ray::ColorAlpha(ray::BLUE, (float) fabs(value)));
        ray::DrawCircleLines(x, y + row, radius, ray::BLUE);

        // Actual value
        std::string valueStr = std::to_string(value);

        ray::DrawText(valueStr.c_str(), x + radius * 2, y + row - 5, 15, ray::WHITE);

        y += radius * 3;
    }
}

void drawCanvas(const NeuralNetwork &nn, Matrix &input, const Matrix &output, int screenWidth, int screenHeight) {
    const int canvasWidth = 280;
    const int canvasHeight = 280;

    const int canvasX = 20;
    const int canvasY = screenHeight / 2;

    // Canvas description
    ray::DrawText("Draw a digit here.\nHold right mouse button to erase.\nPress C to clear all", canvasX, canvasY - 70, 20, ray::GRAY);

    // Draw canvas and check user input
    for (int row = 0; row < 28; row++) {
        for (int col = 0; col < 28; col++) {
            float x = canvasX + col * 10;
            float y = canvasY + row * 10;
            
            // Draw
            ray::DrawRectangle(x, y, 10, 10, ray::ColorBrightness(ray::BLACK, input[row * 28 + col][0]));

            // Check input
            if (ray::CheckCollisionPointRec(ray::GetMousePosition(), ray::Rectangle{x, y, 10, 10})) {
                if (ray::IsMouseButtonDown(ray::MOUSE_BUTTON_LEFT)) {
                    input[row * 28 + col][0] = 1.0;

                    // Shade nearby pixels
                    for (int i = -1; i <= 1; i++)
                        for (int j = -1; j <= 1; j++)
                            if (row + i >= 0 && row + i < 28 && col + j >= 0 && col + j < 28)
                                input[(row + i) * 28 + col + j][0] += 0.1;
                }

                if (ray::IsMouseButtonDown(ray::MOUSE_BUTTON_RIGHT)) {
                    input[row * 28 + col][0] = 0.0;

                    // Erase nearby pixels
                    for (int i = -1; i <= 1; i++)
                        for (int j = -1; j <= 1; j++)
                            if (row + i >= 0 && row + i < 28 && col + j >= 0 && col + j < 28)
                                input[(row + i) * 28 + col + j][0] = 0.00;
                }
            }
        }
    }

    // Outline (when canvas is black you can't see it)
    ray::DrawRectangleLines(canvasX, canvasY, canvasWidth, canvasHeight, ray::GRAY);

    std::string prediction = "Prediction: " + std::to_string(nn.getPrediction(output));

    ray::DrawText(prediction.c_str(), canvasX, canvasY + canvasHeight + 20, 20, ray::GRAY);
}

void runDemo(NeuralNetwork &nn) {
    // Demo parameters (90% accuracy on test data)
    nn.loadParameters("nn-parameters-784-56-28-10.bin");

    // Raylib environment
    const int screenWidth = 1000;
    const int screenHeight = 700;

    ray::InitWindow(screenWidth, screenHeight, "Nano Neural Network");
    ray::SetTargetFPS(60);
    ray::SetTraceLogLevel(0);

    // Free camera
    ray::Camera2D camera = { 0 };
    camera.target = (ray::Vector2){ 0.0f, 0.0f };
    camera.offset = (ray::Vector2){ screenWidth / 2.0f, screenHeight / 2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 1.0f;

    // Initial NN data
    Matrix input = Matrix(784, 1).zero();

    // Main loop
    while (!ray::WindowShouldClose()) {
        // Run Neural Network
        ForwardData result = nn.forwardPropagation(input);

        // Update Camera
        camera.zoom += ((float)ray::GetMouseWheelMove()*0.1f);
        
        if (ray::IsMouseButtonDown(ray::MOUSE_BUTTON_MIDDLE)) {
            ray::Vector2 delta = ray::GetMouseDelta();

            camera.offset.x += delta.x;
            camera.offset.y += delta.y;
        }

        if (ray::IsKeyPressed(ray::KEY_R)) {
            camera.zoom = 1.0f;
            camera.target = (ray::Vector2){ 0.0f, 0.0f };
            camera.offset = (ray::Vector2){ screenWidth / 2.0f, screenHeight / 2.0f };
        }

        // Update input
        if (ray::IsKeyPressed(ray::KEY_C)) {
            input.zero();
        }

        // Start drawing
        ray::BeginDrawing();

            ray::ClearBackground(ray::BLACK);

            // Init camera mode, everything drawn will move with the camera
            ray::BeginMode2D(camera);

                drawWeights(nn, -400, -300);

                drawNeurons(result, -400, 0);

            ray::EndMode2D();

            // Outside of 2D mode everything will be static
            // Info
            ray::DrawText("Use mouse wheel to zoom in-out", 10, 10, 20, ray::GRAY);
            ray::DrawText("Hold mouse middle button to move the camera. Press R to reset", 10, 30, 20, ray::GRAY);

            // Canvas 
            drawCanvas(nn, input, result.output, screenWidth, screenHeight);

        ray::EndDrawing();
    }
}

int main(void) {
    // Init Neural Network
    NeuralNetwork nn;

    nn.addInputLayer(784);

    nn.addHiddenLayer(56);
    nn.addHiddenLayer(28);

    nn.addOutputLayer(10);

    runDemo(nn);

    // InputData train = readCSV("../data/train.csv", 784, 10, 31000); // 31000 total entries
    // InputData test = readCSV("../data/test.csv", 784, 10, 11000);   // 11000 total entries

    // // Train
    // nn.gradientDescent(train.input, train.output, 600, 1e-1);

    // // Test NN accuracy on test data
    // ForwardData result = nn.forwardPropagation(test.input);

    // double acc = nn.calculateAccuracy(result.output, test.output);

    // printf("Accuracy on test data: %.2lf%%\n", 100.0 * acc);

    // nn.saveParameters("nn-parameters-784-56-28-10.bin");

    return 0;
}