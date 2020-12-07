#ifndef NEURON_H
#define NEURON_H
#include <iostream>
#include <math.h>
#include <vector>

typedef unsigned char uchar;

#define learnRate 0.1 // 0.01
#define randWeight (float)(( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.2))

struct nnLayer {
    int in;
    int out;
    float** matrix;
    float* hidden;
    float* error;
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
        //printArray(matrix, in, out);
    }
    void makeHidden(float* input) {
        for (size_t hid = 0; hid < out; hid++) {
            float tmpS = 0;
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
    float sigmoidaDerivate(float val) {
        return (val * (1.0 - val));
    };
    void printArray(float** arr, size_t w, size_t h) {
        std::cout << std::endl << "printArray [][] : " << std::endl;
        for (size_t i = 0; i < w; i++) {
            for (size_t j = 0; j < h; j++)  
                std::cout << arr[i][j] << " ";
            std::cout << std::endl;
        }
    }
    float* getError() {
        return error;
    };
    void calcHidError(float* targets, float** outWeights, int inS, int outS) {
        error = (float*)malloc((inS) * sizeof(float));
        for (int hid = 0; hid < inS; hid++) {
            error[hid] = 0.0;
            for (int ou = 0; ou < outS; ou++)
                error[hid] += targets[ou] * outWeights[hid][ou];

            error[hid] *= sigmoidaDerivate(hidden[hid]);
        }
    };
    void calcOutError(float* targets) {
        error = (float*)malloc((out) * sizeof(float));
        for (int ou = 0; ou < out; ou++)
            error[ou] = (targets[ou] - hidden[ou]) * sigmoidaDerivate(hidden[ou]);

    };
    void updMatrix(float* enteredVal) {
        for (int ou = 0; ou < out; ou++) {
            for (int hid = 0; hid < in; hid++)
                matrix[hid][ou] += (learnRate * error[ou] * enteredVal[hid]);
            matrix[in][ou] += (learnRate * error[ou]);
        } 
    };
};

class Neuron {
public:
    Neuron(std::vector<unsigned int> layers);

    void runThrough(bool ok);
    void backPropagate();
    void train(float* in, float* targ);
    void filling(float* in);
    void printArray(float* arr, size_t s);
    std::pair<int, float> highProbability(float* in);

private:
    struct nnLayer* list; // layers
    int inputNeurons;  // first layer
    int outputNeurons; // last layer
    int layerCount;

    float* inputs;
    float* targets;
};

#endif // NEURON_H