#include <ctime>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits> 
#include <stdint.h>
#include <cstdint>
#include <stdio.h>
#include <Windows.h>
#include <string>
#include <thread>

#include "Neuron.h"
#include "Neuro.h"

typedef unsigned char uchar;
typedef unsigned int uint;

int widht;
int height;
int prog;

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

void getSizeWindows() {
    HANDLE hWndConsole;
    if (hWndConsole = GetStdHandle(-12))
    {
        CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
        if (GetConsoleScreenBufferInfo(hWndConsole, &consoleInfo))
        {
            widht = consoleInfo.srWindow.Right - consoleInfo.srWindow.Left + 1;
            height = consoleInfo.srWindow.Bottom - consoleInfo.srWindow.Top + 1;
        }
        else
            printf("Error: %d\n", GetLastError());
    }
    else
        printf("Error: %d\n", GetLastError());
}

void printProgress(int count_mnist_images_train) {
    getSizeWindows();
    widht -= 19;
    std::string s = "";
    for (int i = 0; i < widht; i++) s += "-";
    std::cout << s.size() << std::endl;
    int one_simbol = count_mnist_images_train / widht;
    int p = 0;
    while (prog < count_mnist_images_train-1) {
        int progress = prog * 100 / count_mnist_images_train;
        printf("\rProcessing (%d%%) [%s]", progress, s.c_str()); //19
        fflush(stdout);
        
        if (count_mnist_images_train * p /92 == prog) {
            if (p < s.size()) {
                s.replace(p, 1, "=");
                p++;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    srand(time(0));

//READ MNIST
    int count_mnist_images_train = -1;
    int count_mnist_images_test = -1;
    int size_mnist_images = 28*28;
    float** image_train = read_mnist_images("..\\train\\train-images.idx3-ubyte", count_mnist_images_train, size_mnist_images);
    int* lable_train = read_mnist_labels("..\\train\\train-labels.idx1-ubyte", count_mnist_images_train);
    float** image_test = read_mnist_images("..\\test\\t10k-images.idx3-ubyte", count_mnist_images_test, size_mnist_images);
    int* lable_test = read_mnist_labels("..\\test\\t10k-labels.idx1-ubyte", count_mnist_images_test);

// CREATE NN
    int input = 28*28;
    int output = 10;
    //                                     first    hiden   exit
    //                                   |-------||--------||--|
    std::vector<unsigned int> size_layers{ 28 * 28, 300, 100, 10}; // size = 3
    //                                              │    │    │             ↑
    //                                             +1   +1   +1             =
    //                                              └────┴────┴─────────────┘
    // Neuron nn(size_layers); // first arhitecture
    NN nn(size_layers); //second architecture (i thk its better)

// GENERATION INPUTS (not use)
    size_t N = 28*28;
    float* first = new float[N];
    float* second = new float[N];
    for (size_t i = 0; i < N; i++){
        first[i] =  (rand() % 98) * 0.01 + 0.01;
        second[i] = (rand() % 98) * 0.01 + 0.01;
    }

// GENERATION OUTPUT - TARGETS
    float** lable_arr_train = new float* [count_mnist_images_train];
    float** lable_arr_test = new float* [count_mnist_images_test];
    for (int i = 0; i < count_mnist_images_train; i++) {
        lable_arr_train[i] = new float[10]();
        lable_arr_train[i][(int)lable_train[i]] = 1.0;
    }
    for (int i = 0; i < count_mnist_images_test; i++) {
        lable_arr_test[i] = new float[10]();
        lable_arr_test[i][(int)lable_test[i]] = 1.0;
    }

// PRINT FIRST LABLE
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 10; j++) std::cout << lable_arr_train[i][j] << " ";
        std::cout << std::endl;
    }

// PRINT HUNDREDTH IMAGE
    std::cout << std::endl << "THIS " << (int)lable_train[100] << std::endl;
    printImg(image_train[100]);
    
// START NN 
    std::cout << std::endl << "-------------------------------------" << std::endl;

    //nn.filling(image_train[333]);
    std::cout << std::endl << "THIS 666" << std::endl;
    nn.filling(image_train[666]);

    std::cout << std::endl << "----------------TRAINIG--------------" << std::endl;
    size_t epoch = 1; // epoch
    count_mnist_images_train = 3000; //rewrite size database

    std::thread thread(printProgress, count_mnist_images_train); // thread for progress bar

    for (size_t i = 0; i < epoch; i++) {
        for (size_t j = 0; j < count_mnist_images_train; j++) {
            prog = j;
            nn.train(image_train[j], lable_arr_train[j]);
        }
    }

    if (thread.joinable()) thread.join();

    std::cout << std::endl << "-----------------RESULT--------------" << std::endl;

    std::cout << std::endl << "THIS " <<(int)lable_train[100] << std::endl;
    nn.filling(image_train[100]);
    std::cout << std::endl << "THIS " << (int)lable_train[200] << std::endl;
    nn.filling(image_train[200]);

    std::cout << std::endl << "---------------TESTING---------------" << std::endl;

    unsigned int current = 0;
    unsigned int error = 0;




    std::cout << std::endl << "---------------THE END---------------" << std::endl;

    return 0;
}