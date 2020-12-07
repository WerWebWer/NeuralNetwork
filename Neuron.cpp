#include "Neuron.h"


Neuron::Neuron(std::vector<unsigned int> layers) {

    int N = layers.size() - 1;
    inputNeurons = layers[0];
    outputNeurons = layers[layers.size() - 1];
    layerCount = N;
    list = (nnLayer*)malloc((N) * sizeof(nnLayer));

    inputs = (float*)malloc((inputNeurons) * sizeof(float));
    targets = (float*)malloc((outputNeurons) * sizeof(float));
    for (int i = 0; i < N; i++) {
        std::cout << "Layer " << i << ": " << layers[i] << " => " << layers[i+1] << std::endl;
        list[i].setIO(layers[i], layers[i + 1]);
    }
}

void Neuron::runThrough(bool stat) {
    list[0].makeHidden(inputs);
    for (int i = 1; i < layerCount; i++)
        list[i].makeHidden(list[i - 1].getHidden());

    if (!stat) {
        std::cout << std::endl << "PROBABILITY: " << std::endl;
        for (int out = 0; out < outputNeurons; out++) {
            float tmp = list[layerCount - 1].hidden[out];
            std::cout << out << ": " << tmp << std::endl;;
        }
        std::cout << std::endl;
        return;
    } else {
        backPropagate();
    }
}

void Neuron::backPropagate() {
    //ERRORS CALC
    list[layerCount - 1].calcOutError(targets);
    for (int i = layerCount - 2; i >= 0; i--)
        list[i].calcHidError(list[i + 1].getError(), list[i + 1].getMatrix(),
            list[i + 1].getInCount(), list[i + 1].getOutCount());
    //UPD WEIGHT
    for (int i = layerCount - 1; i > 0; i--) list[i].updMatrix(list[i - 1].getHidden());
    list[0].updMatrix(inputs);
}


void Neuron::train(float* in, float* targ) {
    inputs = in;
    targets = targ;
    runThrough(true); // => backPropagate();
}

void Neuron::filling(float* in) {
    inputs = in;
    runThrough(false);
}

void Neuron::printArray(float* arr, size_t s) {
    std::cout << std::endl << "printArray [] : " << std::endl;
    for (size_t inp = 0; inp < s; inp++) std::cout << (int)arr[inp];
}

std::pair<int, float> Neuron::highProbability(float* in) {
    inputs = in;
    float max_probability = 0;
    int max_count = -1;
    list[0].makeHidden(inputs);
    for (int i = 1; i < layerCount; i++) list[i].makeHidden(list[i - 1].getHidden());
    for (int i = 0; i < outputNeurons; i++) {
        float tmp = list[layerCount - 1].hidden[i];
        if (max_probability < tmp) {
            max_probability = tmp;
            max_count = i;
        }
    }
    return std::pair<int, float>(max_count,max_probability);
}