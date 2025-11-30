#include "trainer.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <random>


void trainer(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y, float lr, int epochs, bool verbosity, int patience) {
    Eigen::VectorXd x(1);
    Eigen::VectorXd yHat(1);
    Eigen::VectorXd y(1);
    float loss = 0;
    int size = X.rows();
    kan.testing = false;
    int noImprove = 0;
    float bestLoss = std::numeric_limits<float>::max();

    std::vector<int> indices(size);
    for (int i = 0; i < size; ++i)
        indices[i] = i;

    for (int epoch=0; epoch<epochs; ++epoch) {

        std::mt19937 randomEngine(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), randomEngine);

        for (int i : indices) {
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
        if (patience > 0) {
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


void trainTestSplit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double testRatio,
    Eigen::MatrixXd& X_train, Eigen::MatrixXd& X_test, Eigen::MatrixXd& Y_train, Eigen::MatrixXd& Y_test) 
{
    if ((testRatio <= 0) or (testRatio >= 1)) {
        std::cout << "Error: testRatio must be between 0 and 1. The default value of 0.2 is set\n";
        testRatio = 0.2;
    }
    int n = X.rows();

    int numTest = std::round(n * testRatio);
    int numTrain = n - numTest;

    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i)
        indices[i] = i;

    std::mt19937 randomEngine(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), randomEngine);

    X_train.resize(numTrain, X.cols());
    X_test.resize(numTest, X.cols());
    Y_train.resize(numTrain, Y.cols());
    Y_test.resize(numTest, Y.cols());

    for (int i = 0; i < numTrain; ++i) {
        int idx = indices[i];
        X_train.row(i) = X.row(idx);
        Y_train.row(i) = Y.row(idx);
    }

    for (int i = 0; i < numTest; ++i) {
        int idx = indices[numTrain + i];
        X_test.row(i) = X.row(idx);
        Y_test.row(i) = Y.row(idx);
    }
}