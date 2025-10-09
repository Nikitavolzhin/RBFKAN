#include <iostream>
#include <Eigen/Dense>
#include "Header.h"
#include <cstdlib>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

int main()
{
    KAN kan = KAN(3, 0, 1, 1, 1, 2, 2);

    
    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat;
    Eigen::VectorXd y(1);
    for (int i = 0; i < 10000; ++i) {
        x << float((rand() % 101)) / 100;
        yHat = kan.forward(x);
        y = x.array().square();
        kan.backpropagation(y);
        std::cout << (yHat - y).array().square() << std::endl;
    }

    return 0;
}