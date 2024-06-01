#ifndef DEMO_VIEW_H
#define DEMO_VIEW_H

// Need a namespace for raylib because it also uses a struct called Matrix
// This way we can avoid conflicts
namespace ray {
#include "raylib/raylib.h"
}

#include "NeuralNetwork.hpp"

void drawWeights(const NeuralNetwork &nn, int x, int y);

void drawNeurons(const ForwardData &result, int x, int y);

void drawCanvas(const NeuralNetwork &nn, Matrix &input, const Matrix &output, int screenWidth, int screenHeight);

void runDemo(NeuralNetwork &nn);

#endif