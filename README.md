## Neural Network 

by WerWebWer

# Neural Network

This neural network uses [MNIST](http://yann.lecun.com/exdb/mnist/) data and a [Backpropagation](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) algorithm.
By default, the neural network has two hidden layers, dimensions 300 and 100. The error for this neural network ~ 7,10%. Approximate training time = 352192 ms = 352 sec = 6 min.

## New Features!

  - Print images by *0* and *1*

You can also:
  - Save to file wights and read from file txt
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

**Important**  If you don't want the neural network to train every time it starts, then comment out line 19! And then the neural network will read the weights from the file `NN_2020_12_7_21_53_19.txt`

## What's under the box?

#### main.cpp

- Reading MNIST data
- Neural network initialization
- Neural network training or read wights
- Neural network testing

#### Neuro.h/.cpp

Latest architecture implementation: neuron, layer and neural network.

#### Neuron.h/.cpp

Old neural network implementation with Neron class and nnLayer structure.

### TO-DO

 - Add tests
 - Add Trevis

#### History

##### v1.0.3

- Fix save wights to file txt
- Add read from txt file

##### v1.0.2

- Add README
- Add progress bar
- Save wights

##### v1.0.1

- New arhitecture in `Neuro.h/.cpp`
- Print image

##### v1.0.0

- Create Neural network `Neuron.h/.cpp`
