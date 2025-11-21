#include "trainer.h"
#include <iostream>

void trainer(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y, float lr, int epochs, bool verbosity) {
    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat(1);
    Eigen::VectorXd y(1);
    float loss = 0;
    int size = X.rows();
    for (int epoch=0; epoch<epochs; ++epoch) {
        for (int i = 0; i < size; ++i) {
            x = X.row(i).transpose();
            yHat = kan.forward(x);
            y = Y.row(i).transpose();
            kan.backpropagation(y, lr);
            if (verbosity) {
                if (kan.params.loss == "MSE") {
                    loss += (yHat - y).array().square().sum();
                }
                else {
                    loss += (yHat - y).array().abs().sum();
                }
            }
        }
        if (verbosity) {
            std::cout << "Epoch " << epoch;
            std::cout << "; Training loss: " << loss / size << "\n";
            loss = 0;
        }
    }
}
void test(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y) {
    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat(1);
    Eigen::VectorXd y(1);
    float loss = 0;
    int size = X.rows();
    kan.testing = true;
    for (int i = 0; i < size; ++i) {
            x = X.row(i).transpose();
            yHat = kan.forward(x);
            y = Y.row(i).transpose();
            if (kan.params.loss == "MSE") {
                loss += (yHat - y).array().square().sum();
            }
            else {
                loss += (yHat - y).array().abs().sum();
            }
    }
    std::cout << "Loss: " << loss / size << "\n";
    
}