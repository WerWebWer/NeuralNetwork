#include "Neuron.h"


Neuron::Neuron(int N, int first, ...) {

    int* p = &first;
    inputNeurons = 28*28;
    outputNeurons = 10;   
    layerCount = N;
    list = (nnLayer*)malloc((N) * sizeof(nnLayer));

    inputs = (float*)malloc((inputNeurons) * sizeof(float));
    targets = (float*)malloc((outputNeurons) * sizeof(float));
    list[0].setIO(28*28, 300);
    list[1].setIO(300, 100);
    list[2].setIO(100,10);
    //for (int i = 0; i < N; i++) {
    //    std::cout << *p << " " << *(p + 1) << std::endl;
    //    list[i].setIO(*p, *(p+1));
    //    p++;
    //}
}

void Neuron::runThrough(bool stat) {
    list[0].makeHidden(inputs);
    for (int i = 1; i < layerCount; i++)
        list[i].makeHidden(list[i - 1].getHidden());

    if (!stat) {
        std::cout << std::endl << "Feed Forward: " << std::endl;
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
    for (int i = layerCount - 1; i > 0; i--)
        list[i].updMatrix(list[i - 1].getHidden());
    list[0].updMatrix(inputs);
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

void Neuron::printArray(float* arr, size_t s) {
    std::cout << std::endl << "printArray [] : " << std::endl;
    for (size_t inp = 0; inp < s; inp++)
        std::cout << (int)arr[inp];
}