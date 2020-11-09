#include "Neuron.h"

Neuron::Neuron(size_t N, size_t first, ...) {
    size_t* p = &first;
    inputNeurons = *p;
    outputNeurons = *(p+N);   
    layerCount = N;
    list = (nnLayer*)malloc((N) * sizeof(nnLayer));

    inputs = (float*)malloc((inputNeurons) * sizeof(float));
    targets = (float*)malloc((outputNeurons) * sizeof(float));

    for (size_t i = 0; i < N; i++) {
        list[i].setIO(*p, *(p+1));
        p++;
    }
}

void Neuron::runThrough(bool stat) {
    list[0].makeHidden(inputs);
    for (int i = 1; i < layerCount; i++)
        list[i].makeHidden(list[i - 1].getHidden());

    if (true /* !stat */) {
        std::cout << std::endl << "Feed Forward: " ;
        for (int out = 0; out < outputNeurons; out++) {
            std::cout << list[layerCount - 1].hidden[out];
        }
        return;
    } else {
        backPropagate();
    }
}

void Neuron::backPropagate() {
    // TODO
}

void Neuron::train(float* in, float* targ) {
    inputs = in;
    targets = targ;
    runThrough(true);
}

void Neuron::filling(float* in) {
    inputs = in;
    runThrough(false);
}

void Neuron::printArray(float* arr, int s) {
    std::cout << std::endl << "printArray : " << std::endl;
    for (int inp = 0; inp < s; inp++)
        std::cout << arr[inp];
}