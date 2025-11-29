#include "trainer.h"
#include <iostream>
#include <cmath>
#include <limits>

void trainer(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y, float lr, int epochs, bool verbosity, int patience) {
    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat(1);
    Eigen::VectorXd y(1);
    float loss = 0;
    int size = X.rows();
    kan.testing = false;
    int noImprove = 0;
    float bestLoss = std::numeric_limits<int>::max();
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
        if (patience != 0) {
            if (bestLoss > loss / size) {
                bestLoss = loss / size;
                noImprove = 0;
            }
            else {
                noImprove++;
                if (patience <= noImprove) {
                    std::cout << "Max patience is reached. Training is over.\n";
                    break;
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