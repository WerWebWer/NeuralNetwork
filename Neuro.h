#ifndef NEURO_LAYER_NN_H
#define NEURO_LAYER_NN_H
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <ctime>
#include <string>

#define learnRate 0.1 // better 0.01 ???
#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(size,-0.2))
#define sigmoidsFormula (1.0 / (1.0 + exp(-val)))
#define sigmoidaDerivateFormula (val * (1.0 - val))

class Neuro {
public:
    Neuro();
    void init(unsigned int in);
    float getWight(unsigned int i) { return wights[i]; };
    unsigned int getSize() { return size; };
    void setRandWights() { for (size_t i = 0; i < size; i++) wights[i] = randWeight; };
    void setAddWight(unsigned int i, float val);

private:
    unsigned int size;
    float* wights;
};

class Layer {
public:
    Layer();
    Neuro* getMatrix() { return matrix; };
    unsigned int getInCount() { return in; };
    unsigned int getOutCount() { return out; };
    float* getError() { return error; };
    float* getHidden() { return hidden; };
    float getHid(unsigned int i) { return hidden[i]; };
    float sigmoida(float val) { return sigmoidsFormula; };
    float sigmoidaDerivate(float val) { return sigmoidaDerivateFormula; };
    void setIO(unsigned int in, unsigned int out);
    void makeHidden(float* input);
    void calcOutError(float* targets);
    void calcHidError(float* targets, Neuro* Weights, unsigned int inS, unsigned int outS);
    void updMatrix(float* enteredVal);
    std::pair<int, int> getSize() { return std::pair<int, int>(in, out); };
    void setMatrix();
private:
    unsigned int in; // this layer
    unsigned int out; // next layer
    Neuro* matrix; // weight
    float* hidden; // intermediate "pseudo layer"
    float* error; // collecting errors from back Propagate
};

class NN {
public:
    NN(std::vector<unsigned int> layers);
    void filling(float* in); // init layers
    void train(float* in, float* targ); // => backPropagate()
    void backPropagate(); 
    std::pair<int, float> highProbability(float* in); // find out what it is
    void saveNN();
private:
    Layer* list; // layers
    int inputNeurons;  // first layer
    int outputNeurons; // last layer
    int layerCount;  // count layers

    float* inputs; // input data (image)
    float* targets; // correct output (array of probabilities)

};
#endif // NEURO_LAYER_NN_H