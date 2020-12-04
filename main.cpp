#include <ctime>

#include "Neuron.h"
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits> 
#include <stdint.h>
#include <cstdint>

typedef unsigned char uchar;
typedef unsigned int uint;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float convert(uchar value){
    return (float)value / 255.0;
}

void printImg(float* img) {
    for (int i = 0; i < 28*28; i++) {
        if (i % 28 == 0) std::cout << std::endl;
        std::cout << (img[i] > (float)1e-10) ? 1 : 0;
    }
    std::cout << std::endl;
}

float** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {
    // convert variable unsigned char to float in the range from 0 to 1
    auto uchar2float = [](uchar var) { 
        float res;
        res = (float)var / 255.0;
        return res;
    };
    // convert array unsigned char to float in the range from 0 to 1
    auto uchar2float_array = [](uchar* var, uint size) {
        float* res = new float[size];
        for (uint i = 0; i < size; i++) res[i] = (float)var[i] / 255.0;
        return res;
    };

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset_uchar = new uchar* [number_of_images];
        float** _dataset_float = new float* [number_of_images];
        for (int i = 0; i < number_of_images; i++) {
            _dataset_uchar[i] = new uchar[image_size];
            _dataset_float[i] = new float[image_size];
            file.read((char*)_dataset_uchar[i], image_size);
            for (int j = 0; j < image_size; j++) _dataset_float[i][j] = uchar2float(_dataset_uchar[i][j]);

        }
        return _dataset_float;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`");
    }
}

int* read_mnist_labels(std::string full_path, int& number_of_labels) {
    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        int* _dataset_int = new int[number_of_labels];
        uchar* _dataset_uchar = new uchar[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset_uchar[i], 1);
            _dataset_int[i] = (int)_dataset_uchar[i];
        }
        return _dataset_int;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`");
    }
}

int main(int argc, char* argv[]) {
    srand(time(0));
    int count_mnist_images = 1;
    int size_mnist_images = 28*28;

    float** image = read_mnist_images("..\\train\\train-images.idx3-ubyte", count_mnist_images, size_mnist_images);
    int* lable = read_mnist_labels("..\\train\\train-labels.idx1-ubyte", count_mnist_images);

    size_t epoch = 1; // epoch
    // count_mnist_images = 30000;

// CREATE NN
    int input = 28*28;
    int output = 10;
    //                         first  hiden     exit
    //                        |-----||--------||-----|
    Neuron* nn = new Neuron(3, input, 300, 100, output);
    //                      ↑          │    │     │
    //                      =         +1   +1    +1
    //                      └──────────┴────┴─────┘

// GENERATION INPUTS
    size_t N = 28*28;
    float* first = new float[N];
    float* second = new float[N];
    for (size_t i = 0; i < N; i++){
        first[i] =  (rand() % 98) * 0.01 + 0.01;
        second[i] = (rand() % 98) * 0.01 + 0.01;
        // std::cout << first[i] << " " << second[i] << std::endl;
    }

// GENERATION OUTPUT - TARGETS

    float** lable_arr = new float* [count_mnist_images];
    for (int i = 0; i < count_mnist_images; i++) {
        lable_arr[i] = new float[10]();
        lable_arr[i][(int)lable[i]] = 1.0;
    }

// PRINT FIRST LABLE AND IMAGE
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 10; j++) std::cout << lable_arr[i][j] << " ";
        std::cout << std::endl;
    }
    printImg(image[100]);
    

// START NN 
    std::cout << std::endl << "-------------------------------------" << std::endl;
    nn->filling(image[333]);
    nn->filling(image[666]);


    std::cout << std::endl << "----------------TRAINIG--------------" << std::endl;

    for (size_t i = 0; i < epoch; i++) {
        for (size_t j = 0; j < count_mnist_images; j++) {
            //if (j % 100 == 0 && j != 0) std::cout << "Done: " << j << std::endl;
            nn->train(image[j], lable_arr[j]);
        }
    }

    std::cout << std::endl << "-----------------RESULT--------------" << std::endl;

    std::cout << std::endl << "THIS " <<(int)lable[100] << std::endl;
    nn->filling(image[100]);
    std::cout << std::endl << "THIS " << (int)lable[200] << std::endl;
    nn->filling(image[200]);


    std::cout << std::endl << "---------------THE END---------------" << std::endl;

    return 0;
}