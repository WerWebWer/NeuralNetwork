//#include "Neuro.h"
//
//Neuro::Neuro() {
//
//}
//
//float Neuro::sigmoida(float val) {
//    return (1.0 / (1.0 + exp(-val)));
//};
//
//void Neuro::init(unsigned int coms) {
//    countComm = coms;
//    comm = (float*)malloc((coms) * sizeof(float));
//};
//
//void Neuro::communication() {
//    for(size_t i = 0; i < countComm; i++)
//        weight[i] = randWeight;
//};
//
//Layer::Layer() {
//
//}
//void Layer::setIO(unsigned int in, unsigned int out) {
//    size = in;
//    next = out;
//    matrix = (Neuro*)malloc((size) * sizeof(Neuro));
//    for (size_t inp = 0; inp < size; inp++) {
//        matrix[inp].init(next);
//    }
//    for (size_t inp = 0; inp < in + 1; inp++) {
//        for (size_t outp = 0; outp < out; outp++) {
//            matrix[inp][outp] = randWeight;
//        }
//    }
//}
//
//NN::NN(unsigned int N, unsigned int first, ...) {
//    unsigned int* p = &first;
//    inputNeurons = *p;
//    outputNeurons = *(p + N);
//    layerCount = N;
//    list = (Layer*)malloc((N) * sizeof(Layer));
//
//    inputs = (float*)malloc((inputNeurons) * sizeof(float));
//    targets = (float*)malloc((outputNeurons) * sizeof(float));
//
//    for (size_t i = 0; i < N; i++, p++) {
//        list[i].setIO(*p, *(p + 1));
//    }
//}