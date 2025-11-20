#include "trainer.h"
#include <iostream>

void trainer(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y, bool verbosity) {
    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat(1);
    Eigen::VectorXd y(1);
    float loss = 0;
    int size = X.rows();
    for (int i = 0; i < size; ++i) {
        x = X.row(i).transpose();
        yHat = kan.forward(x);
        y = Y.row(i).transpose();
        kan.backpropagation(y, 0.1);
        if (verbosity) {
            if (kan.params.loss == "MSE") {
                loss += (yHat - y).array().square().sum();
            }
            else {
                loss += (yHat - y).array().abs().sum();
            }
            if ((i+1)% kan.batchSize == 0) {
                std::cout << "Batch " << int((i + 1) / kan.batchSize);
                std::cout << "; Training loss: " << loss / kan.batchSize << "\n";
                loss = 0;
            }
        }
    }
}