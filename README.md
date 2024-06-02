# Nano Neural Network

## Description

This is a simple Multilayer Perceptron (MLP) Neural Network implementation in C++. I did this project for fun because I wanted to learn how NNs work and this is one of the simplest to implement.

Everything is implemented from scratch, including the linear algebra needed for Matrix operations.

A real time visualization and interactive demo is possible on trained networks.

This code is not meant to be blazingly fast, though some optimizations were made here and there.
As far as I've tested, this implementation is 1.5x to 2.5x slower than a pure numpy implementation (no TensorFlow).

This project was roughly based on [this video](https://www.youtube.com/watch?v=w8yWXqWQYmU), I would also recommend [3B1B playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=euRyhi6ECpi-81Ri) on NNs.

## Training dataset and accuracy

This Neural Network was trained on part of the MNIST handwritten digits database:

```sh
Reading train.csv ... Done!
Reading test.csv ... Done!
Iteration: 0  Loss (MSE): 0.140818  Accuracy: 7.44%
Iteration: 10  Loss (MSE): 0.077290  Accuracy: 36.97%
Iteration: 20  Loss (MSE): 0.062988  Accuracy: 51.55%
Iteration: 30  Loss (MSE): 0.052713  Accuracy: 60.80%
Iteration: 40  Loss (MSE): 0.046117  Accuracy: 66.21%
Iteration: 50  Loss (MSE): 0.041670  Accuracy: 69.64%
Iteration: 60  Loss (MSE): 0.038439  Accuracy: 72.27%
...
Iteration: 560  Loss (MSE): 0.014004  Accuracy: 90.68%
Iteration: 570  Loss (MSE): 0.013891  Accuracy: 90.74%
Iteration: 580  Loss (MSE): 0.013780  Accuracy: 90.81%
Iteration: 590  Loss (MSE): 0.013673  Accuracy: 90.88%
Iteration: 599  Loss (MSE): 0.013578  Accuracy: 90.95%
Accuracy on test data: 89.85%
```

- The accuracy on training data was 90.95% for 31000 entries.
- The accuracy on test data was 89.85% for 11000 entries.

## How to run the demo (or train)

You can find compiled binaries and pre-trained data in the **Releases** section for Windows and Linux. 

- Run the program without any arguments to start the visualization demo. 
- Run the program with any argument to start the training process.

**NOTE:** The compiled binary for Linux uses GLIBC 2.35.
You can clone and compile the project locally if you're having problems with incompatible versions.

## How to compile

Compiled libraries for *raylib* are provided for Windows and Linux systems, you don't need to worry about compiling *raylib*.
The Makefile will select the appropriate library automatically.

To compile and run:

```sh
make
make run
```

### Linux requirements 

You should be good to go on most situations. However you may need to install *libgomp* if the program fails to find *libgomp.so*:

```sh
sudo apt-get install libgomp1
```

### Windows requirements

You need [MinGW](https://code.visualstudio.com/docs/cpp/config-mingw) to compile this project since it uses OpenMP pragmas. 

## Visualization demo

The main feature of this project is an interactive demo, allowing a real time peek at the inner workings of the network:

- You can see the values of every neuron and weights.
- You can see how neurons react to your input.
- You can see the current prediction for your input.

A canvas on the lower left of the screen allows the user to draw a number.

The numbers on the training dataset are big and occupy most of the grid.
You will need to draw big numbers with thick lines.

![demo_gif](https://github.com/probablygab/nano-nn/assets/96994614/472ea101-122d-4fdd-a75e-56b1df0456d6)

![group1](https://github.com/probablygab/nano-nn/assets/96994614/8702aa09-a2e4-4dd7-910f-f769302bd496)
![group2](https://github.com/probablygab/nano-nn/assets/96994614/86c5f9fd-27d1-4e05-ad61-f30e6e30c96c)

### Shortcomings

The MNIST dataset is very popular, but it is quite uniform when it comes to number configurations. 
That is, the network wasn't trained on numbers of different sizes or tilted numbers. 
If you draw a number in weird ways, the network may not recognize it.

## Customizing the Neural Network

You can play around with the layers and train the model again, my best accuracy was 90% with the current model. Maybe you can beat me :)

```C
NeuralNetwork nn;

nn.addInputLayer(INPUT_SIZE);

// Customize here
nn.addHiddenLayer(10);

nn.addOutputLayer(OUTPUT_SIZE);
```

If you want to use this Neural Network implementation on custom datasets, you will need to create a custom readCSV function to load the input and output data.
You will also need to modify some #defines. This assumes the network will be used for classification. No other changes are necessary.
