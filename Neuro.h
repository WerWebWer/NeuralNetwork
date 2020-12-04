//#ifndef NEURO_H
//#define NEURO_H
//#include <iostream>
//#include <math.h>
//
//#define learnRate 0.1 // 0.01
//#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.2))
//
//class Neuro {
//public:
//    Neuro();
//    void setWeight(float _weight);
//    void init(unsigned int coms);
//    void communication();
//    float sigmoida(float val);
//
//private:
//    float weight;
//    float* comm;
//    unsigned int countComm;
//};
//
//class Layer {
//public:
//    Layer();
//    void setIO(unsigned int next) {}
//private:
//    unsigned int size; // this layer
//    unsigned int next; // next layer
//    Neuro* matrix; // weight
//};
//
//class NN {
//public:
//    NN(unsigned int N, unsigned int first, ...);
//private:
//    Layer* list; // layers
//    int inputNeurons;  // first layer
//    int outputNeurons; // last layer
//    int layerCount;
//
//    float* inputs;
//    float* targets;
//
//};
//#endif // NEURO_H