#include <ctime>

#include "Neuron.h"

int main(int argc, char* argv[]) {
    srand(time(0));

    size_t epoch = 100; // epoch

// CREATE NN
    size_t input = 100;
    size_t output = 2;
    //                         first   hiden    exit
    //                        |-----||--------||-----|
    Neuron* nn = new Neuron(4, input, 20, 6, 3, output);

// GENERATION INPUTS
    size_t N = 100;
    float* first = new float[N];
    float* second = new float[N];
    for (size_t i = 0; i < N; i++) 
        first[i] = (rand() % 98) * 0.01 + 0.01;

    for (size_t i = 0; i < N; i++)
        second[i] = (rand() % 98) * 0.01 + 0.01;

// GENERATION OUTPUT - TARGETS
    float* tar1 = new float[2];
    float* tar2 = new float[2];

    tar1[0] = 0.01;
    tar1[1] = 0.99;

    tar2[0] = 0.99;
    tar2[1] = 0.01;

// START NN 
    std::cout << std::endl << "-------------------------------------" << std::endl;
    nn->filling(first);
    nn->filling(second);

    int i = 0;
    std::cout << std::endl << "----------------TRAINIG--------------" << std::endl;
    for (size_t i = 0; i < epoch; i++) {
        nn->train(first, tar1);
        nn->train(second, tar2);
    }

    std::cout << std::endl << "----------------RESULT---------------" << std::endl;
    nn->filling(first);
    nn->filling(second);


    std::cout << std::endl << "---------------THE END---------------" << std::endl;

    return 0;
}