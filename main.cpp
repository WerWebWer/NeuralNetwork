#include <ctime>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits> 
#include <cstdint>
#include <stdio.h>
#include <Windows.h>
#include <string>
#include <thread>

#include "Neuron.h"
#include "Neuro.h"

typedef unsigned char uchar;
typedef unsigned int uint;

#define TRAINING true // comment out so as not to train again

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

float** read_mnist_images(std::string full_path, uint& number_of_images, uint& image_size) {
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

int* read_mnist_labels(std::string full_path, uint& number_of_labels) {
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

std::pair<int, int> getSizeWindows() {
    HANDLE hWndConsole;

    int widht = 0; // size console
    int height = 0;

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
    return std::pair<int, int>(widht, height);
}

void printProgress(uint max, uint *prog) {
    std::pair<int, int> size = getSizeWindows();
    uint widht = size.first;
    widht -= 19;
    std::string s = "";
    for (uint i = 0; i < widht; i++) s += "-";
    uint p = 0;
    while (*prog < max-1) {
        uint progress = *prog * 100 / max;
        printf("\rProcessing (%d%%) [%s]", progress, s.c_str()); //19
        fflush(stdout);
        
        if (max * p / widht == (*prog)) {
            if (p < s.size()) {
                s.replace(p, 1, "=");
                p++;
            }
        }
    }
    s = s.substr(0, s.size() - 1);
    printf("\rProcessing (%d%%) [%s]\n", 100, s.c_str());
}

int main(int argc, char* argv[]) {
    srand(time(0));

    uint prog = 0; //progress bar

//READ MNIST
    uint size_mnist_images = 28*28; // image
    uint count_mnist_images_train = 60000; // size train
    uint count_mnist_images_test = 10000; //size test
    float** image_train = read_mnist_images("..\\train\\train-images.idx3-ubyte", count_mnist_images_train, size_mnist_images);
    int* lable_train = read_mnist_labels("..\\train\\train-labels.idx1-ubyte", count_mnist_images_train);
    float** image_test = read_mnist_images("..\\test\\t10k-images.idx3-ubyte", count_mnist_images_test, size_mnist_images);
    int* lable_test = read_mnist_labels("..\\test\\t10k-labels.idx1-ubyte", count_mnist_images_test);

// GENERATION OUTPUT - TARGETS
    float** lable_arr_train = new float* [count_mnist_images_train];
    float** lable_arr_test = new float* [count_mnist_images_test];
    for (uint i = 0; i < count_mnist_images_train; i++) {
        lable_arr_train[i] = new float[10]();
        lable_arr_train[i][(int)lable_train[i]] = 1.0;
    }
    for (uint i = 0; i < count_mnist_images_test; i++) {
        lable_arr_test[i] = new float[10]();
        lable_arr_test[i][(int)lable_test[i]] = 1.0;
    }
#ifdef TRAINING

// CREATE NN
    uint input = 28*28;
    uint output = 10;
    //                                    first    hiden    exit
    //                                   |-----||--------||------|
    std::vector<unsigned int> size_layers{input, 300, 100, output }; // size = 3
    //                                            │    │     │              ↑
    //                                           +1   +1    +1              =
    //                                            └────┴─────┴──────────────┘

    //Neuron nn(size_layers); // first arhitecture
    NN nn(size_layers); //second architecture (i thk its better)

// GENERATION INPUTS (not use)
    uint N = 28*28;
    float* first = new float[N];
    float* second = new float[N];
    for (size_t i = 0; i < N; i++){
        first[i] =  (rand() % 98) * 0.01 + 0.01;
        second[i] = (rand() % 98) * 0.01 + 0.01;
    }

// PRINT FIRST LABLE

    //for (int i = 0; i < 1; i++) {
    //    for (int j = 0; j < 10; j++) std::cout << lable_arr_train[i][j] << " ";
    //    std::cout << std::endl;
    //}

// PRINT HUNDREDTH IMAGE
    std::cout << std::endl << "THIS " << (int)lable_train[100] << std::endl;
    printImg(image_train[100]);
    
// START NN 
    std::cout << std::endl << "-------------------------------------" << std::endl;

    std::cout << std::endl << "THIS IS " << lable_train[100] << " (100)"; //1000 from database
    nn.filling(image_train[100]);

    std::cout << std::endl << "----------------TRAINIG--------------" << std::endl;
    std::cout << std::endl;

    auto begin = std::chrono::steady_clock::now();

    uint epoch = 1;               // epoch
    //count_mnist_images_train = 300; //rewrite size train database

    std::thread thread_train(printProgress, count_mnist_images_train, &prog); // thread for progress bar
    
    for (uint i = 0; i < epoch; i++) {
        for (uint j = 0; j < count_mnist_images_train; j++) {
            prog = j;
            nn.train(image_train[j], lable_arr_train[j]);
        }
    }

    if (thread_train.joinable()) thread_train.join();

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << std::endl << "The time: " << elapsed_ms.count() << " ms\n";

    std::cout << std::endl << "----------------RESULT---------------" << std::endl;

    std::cout << std::endl << "THIS " <<(int)lable_train[100] << " (100)" << std::endl;
    nn.filling(image_train[100]);

// SAVE WIGHTS TO FILE
    std::cout << std::endl << "------------------SAVE---------------" << std::endl;
    nn.saveNN();

#else //TRAINING
    std::string name = "..\\NN_784_300_100_10.txt";
    NN nn(name);

#endif //TRAINING

// TEST NN 

    std::cout << std::endl << "----------------TESTING--------------" << std::endl;
    std::cout << std::endl;
    unsigned int current = 0;
    unsigned int error = 0;
    std::vector<std::pair<int, float>> errors;
    //count_mnist_images_test = 1000; //rewrite size test database 

    std::thread thread_test(printProgress, count_mnist_images_test, &prog); // thread for progress bar

    for (unsigned int i = 0; i < count_mnist_images_test; i++) {
        std::pair<int, float> res = nn.highProbability(image_test[i]);
        prog = i;
        if (res.first != lable_test[i]) {
            error++;
            errors.push_back(res);
        } else {
            current++;
        }
    }

    if (thread_test.joinable()) thread_test.join();

    std::cout << "TOTAL error: " << (float)error * 100.0 / (float)count_mnist_images_test << "%" << std::endl;

    std::cout << std::endl << "---------------THE END---------------" << std::endl;

    return 0;
}