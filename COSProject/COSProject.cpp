#include <iostream>
#include <Eigen/Dense>
#include "Header.h"
#include <cstdlib>
#include <unsupported/Eigen/CXX11/Tensor>

void trainer(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y) {

    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat(1);
    Eigen::VectorXd y(1);
    int size = X.rows();
    for (int i = 0; i < size; ++i) {
        x = X.row(i).transpose();
        yHat = kan.forward(x);
        y = Y.row(i).transpose();
        kan.backpropagation(y, 0.1);
        std::cout << (yHat - y).array().square() << std::endl;
    }
}

int main()
{

    KAN kan = KAN(3, 0, 1, 1, 1, 3, 3);
    
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(1000, 1);
    Eigen::MatrixXd Y = X.array().square();
    trainer(kan, X, Y);

    return 0;
}