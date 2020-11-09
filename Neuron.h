#ifndef NEURON_H
#define NEURON_H
#include <iostream>
#include <math.h>

#define learnRate 0.1 // 0.01
#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.2))

struct nnLayer {
    int in;
    int out;
    float** matrix;
    float* hidden;
    int getInCount() { return in; }
    int getOutCount() { return out; }
    float** getMatrix() { return matrix; }
    void setIO(int input, int output) {
        in = input;
        out = output;
        hidden = (float*)malloc((out) * sizeof(float));
        matrix = (float**)malloc((in + 1) * sizeof(float*));
        for (size_t inp = 0; inp < in + 1; inp++) {
            matrix[inp] = (float*)malloc(out * sizeof(float));
        }
        for (size_t inp = 0; inp < in + 1; inp++) {
            for (size_t outp = 0; outp < out; outp++) {
                matrix[inp][outp] = randWeight;
            }
        }
    }
    void makeHidden(float* input) {
        for (size_t hid = 0; hid < out; hid++) {
            float tmpS = 0.0;
            for (size_t inp = 0; inp < in; inp++)
                tmpS += input[inp] * matrix[inp][hid];
            tmpS += matrix[in][hid];
            hidden[hid] = sigmoida(tmpS);
        }
    };
    float* getHidden() {
        return hidden;
    };
    float sigmoida(float val) {
        return (1.0 / (1.0 + exp(-val)));
    };
};

class Neuron {
public:
    Neuron(size_t N, size_t first, ...);

    void runThrough(bool ok);
    void backPropagate();
    void train(float* in, float* targ);
    void filling(float* in);
    void printArray(float* arr, int s);

private:
    struct nnLayer* list; // layers
    int inputNeurons;  // first layer
    int outputNeurons; // last layer
    int layerCount;

    float* inputs;
    float* targets;
};

#endif // NEURON_H