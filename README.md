## Neural Network 

by WerWebWer

# Neural Network

This neural network uses [MNIST](http://yann.lecun.com/exdb/mnist/) data and a backpropogation algorithm.
By default, the neural network has two hidden layers, dimensions 300 and 100. The error for this neural network = 7,15%

## New Features!

  - Print images by *0* and *1*

You can also:
  - Save to file wights
  - By hand in the code change the number of layers and their dimension
  - Rewrite all formulas in `#define`

## How to start?

Clone the repository and run the following commands

```sh
$ mkdir build && cd build
$ cmake ..
$ cmake --build . --config RELEASE
```

Then run `./build/Release/NN.exe`

## What's under the box?

#### main.cpp

- Reading MNIST data
- Neural network initialization
- Neural network training
- Neural network testing

#### Neuro.h/.cpp

Latest architecture implementation: neuron, layer and neural network.

#### Neuron.h/.cpp

Old neural network implementation with Neron class and Layer structure.

### TO-DO

 - Read wights from file
 - Add tests
 - Add Trevis

#### History

##### v1.0.2

- Add README
- Add progress bar
- Save wights

##### v1.0.1

- New arhitecture in `Neuro.h/.cpp`
- Print image

##### v1.0.0

- Create Neural network `Neuron.h/.cpp`
