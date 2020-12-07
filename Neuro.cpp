#include "Neuro.h"

void Neuro::init(unsigned int _size) {
    size = _size;
    wights = (float*)malloc(_size * sizeof(float));
};

void Neuro::setAddWight(unsigned int i, float val) {
    wights[i] += val;
}

void Neuro::setWight(unsigned int i, float val) {
    wights[i] = val;
}

void Layer::setIO(unsigned int _in, unsigned int _out) {
    in = _in;
    out = _out;
    matrix = (Neuro*)malloc((in + 1) * sizeof(Neuro));
    hidden = (float*)malloc((out) * sizeof(float));
    for (size_t i = 0; i < in + 1; i++) {
        matrix[i].init(out);
    }
    for (size_t i = 0; i < in + 1; i++) {
        matrix[i].setRandWights();
    }
}

void Layer::makeHidden(float* input) {
    for (size_t hid = 0; hid < out; hid++) {
        float tmpS = 0;
        for (size_t inp = 0; inp < in; inp++)
            tmpS += input[inp] * matrix[inp].getWight(hid);
        tmpS += matrix[in].getWight(hid);
        hidden[hid] = sigmoida(tmpS);
    }
}

void Layer::calcOutError(float* targets) {
    error = (float*)malloc((out) * sizeof(float));
    for (int ou = 0; ou < out; ou++)
        error[ou] = (targets[ou] - hidden[ou]) * sigmoidaDerivate(hidden[ou]);
}

void Layer::calcHidError(float* targets, Neuro* Weights, unsigned int inS, unsigned int outS) {
    error = (float*)malloc((inS) * sizeof(float));
    for (int hid = 0; hid < inS; hid++) {
        error[hid] = 0.0;
        for (int ou = 0; ou < outS; ou++)
            error[hid] += targets[ou] * Weights[hid].getWight(ou);
        error[hid] *= sigmoidaDerivate(hidden[hid]);
    }
}

void Layer::updMatrix(float* enteredVal) {
    for (int ou = 0; ou < out; ou++) {
        for (int hid = 0; hid < in; hid++)
            matrix[hid].setAddWight(ou, learnRate * error[ou] * enteredVal[hid]);
        matrix[in].setAddWight(ou, learnRate * error[ou]);
    }
}

void Layer::setMatrix(int w, int h, float wight) {
    matrix[w].setWight(h, wight);
}


void Layer::setHidden(int i, float val) {
    hidden[i] = val;
}

NN::NN(std::vector<unsigned int> layers) {
    int N = layers.size() - 1;
    inputNeurons = layers[0];
    outputNeurons = layers[layers.size() - 1];
    layerCount = N;
    list = (Layer*)malloc((N) * sizeof(Layer));

    inputs = (float*)malloc((inputNeurons) * sizeof(float));
    targets = (float*)malloc((outputNeurons) * sizeof(float));
    for (int i = 0; i < N; i++) {
        std::cout << "Layer " << i << ": " << layers[i] << " => " << layers[i + 1] << std::endl;
        list[i].setIO(layers[i], layers[i + 1]);
    }
}

NN::NN(std::string path) {
    readNN(path);
}

void NN::filling(float* in) {
    inputs = in;
    list[0].makeHidden(inputs);
    for (int i = 1; i < layerCount; i++)
        list[i].makeHidden(list[i - 1].getHidden());
    std::cout << std::endl << "PROBABILITY: " << std::endl;
    for (int out = 0; out < outputNeurons; out++) {
        float tmp = list[layerCount - 1].getHid(out);
        std::cout << out << ": " << tmp << std::endl;;
    }
    return;
}

void NN::train(float* in, float* targ) {
    inputs = in;
    targets = targ;
    list[0].makeHidden(inputs);
    for (int i = 1; i < layerCount; i++)
        list[i].makeHidden(list[i - 1].getHidden());
    backPropagate();
}

void NN::backPropagate() {
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

std::pair<int, float> NN::highProbability(float* in) {
    inputs = in;
    float max_probability = 0;
    int max_count = -1;
    list[0].makeHidden(inputs);
    for (int i = 1; i < layerCount; i++) 
        list[i].makeHidden(list[i - 1].getHidden());
    for (int i = 0; i < outputNeurons; i++) {
        float tmp = list[layerCount - 1].getHid(i);
        if (max_probability < tmp) {
            max_probability = tmp;
            max_count = i;
        }
    }
    return std::pair<int, float>(max_count, max_probability);
}

void NN::saveNN() {
    std::time_t t = std::time(0); // get time now
    std::tm* now = std::localtime(&t);

    std::string name = "../NN_" + std::to_string(now->tm_year + 1900) +"_" +
        std::to_string(now->tm_mon + 1) + "_" + std::to_string(now->tm_mday) + "_" +
        std::to_string(now->tm_hour) + "_" + std::to_string(now->tm_min) + "_" + 
        std::to_string(now->tm_sec) + ".txt";

    std::ofstream out (name);
    out << layerCount << std::endl;
    for (int i = 0; i < layerCount; i++)
        out  << list[i].getSize().first << " " << list[i].getSize().second << std::endl;
    for (int i = 0; i < layerCount; i++) {
        for (int hid = 0; hid < list[i].getInCount(); hid++)
            for (int ou = 0; ou < list[i].getOutCount(); ou++) {
                float tmp = list[i].getMatrix()[hid].getWight(ou);
                out << tmp << " ";
            }
        out << std::endl;
    }
    //for (int i = 0; i < layerCount; i++) {
    //    for (int hid = 0; hid < list[i].getOutCount(); hid++)
    //            out << list[i].getHidden()[hid] << " ";
    //    out << std::endl;
    //}
}

void NN::readNN(std::string path) {
    std::ifstream file(path, std::ios::out);

    if (file.is_open()) {
        file >> layerCount;
        list = (Layer*)malloc((layerCount) * sizeof(Layer));
        int tmp = -1;
        int in = 0, out = 0;
        for (int i = 0; i < layerCount; i++) {
            file >> in >> out;
            if (i == 0) tmp = in;
            list[i].setIO(in, out);
        }
        inputs = (float*)malloc((tmp) * sizeof(float));
        targets = (float*)malloc((out) * sizeof(float));
        float wight = 0;
        for (int i = 0; i < layerCount; i++)
            for (int hid = 0; hid < list[i].getInCount(); hid++)
                for (int ou = 0; ou < list[i].getOutCount(); ou++) {
                    file >> wight;
                    list[i].setMatrix(hid, ou, wight);
                }
        //for (int i = 0; i < layerCount; i++)
        //    for (int hid = 0; hid < list[i].getOutCount(); hid++) {
        //        file >> wight;
        //        list[i].setHidden(hid, wight);
        //    }
    } else {
        throw std::runtime_error("Unable to open file `" + path + "`");
    }
}
