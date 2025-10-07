#include <iostream>
#include <Eigen/Dense>
#include "Header.h"
#include <cstdlib>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

int main()
{
    /*
    Eigen::VectorXd v1(2);
    v1 << 1, 2;
    RBF func = RBF(0, 1, 5);
    Eigen::VectorXd M = func.forward(v1);
    Layer l = Layer(2*5, 3);
    //std::cout << l.forward(M) << std::endl;
    //std::cout << func.dRBF;
    */
    // Experiment: backpropogating one layer
    
    KAN kan = KAN(3, 0, 1, 1, 1, 3, 2);

    
    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat;
    Eigen::VectorXd y(1);
    for (int i = 0; i < 1; ++i) {
        x << float((rand() % 101)) / 100;
        yHat = kan.forward(x);
        y = x.array().square();
        //kan.backpropagation(yHat, y);

        std::cout << (yHat - y).array().square() << std::endl;
    }

    /*
    RBF rbf = RBF(0, 1, 3);
    Eigen::VectorXd activation;

    Layer layer = Layer(3, 1);
    for (int i = 0; i < 2000; ++i) {
        x << float((rand()%101)) / 100;
        activation = rbf.forward(x);
        yHat = layer.forward(activation);
        y = x.array().square();
        //std::cout << x << " " << yHat << std::endl;
        std::cout << (yHat - y).array().square() << std::endl;
        layer.weights -= 0.2*(yHat - y) * activation.transpose();
    }
    */
    return 0;
}