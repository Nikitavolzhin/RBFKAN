#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <unsupported/Eigen/CXX11/Tensor>
#include "classes.h"
#include "io.h"
#include "trainer.h"
#include "unitTesting.h"

int main()
{
    runTests();
    Eigen::MatrixXd mat = readCSV("data_1.csv");
    Eigen::MatrixXd X = mat.leftCols(2);
    Eigen::MatrixXd Y = mat.rightCols(1);
    Eigen::MatrixXd X_train, X_test, Y_train, Y_test;

    trainTestSplit(X, Y, 0.2,   
        X_train, X_test,
        Y_train, Y_test);
    config params = readConfig("config.json");
    
    KAN* kan = dynamic_cast<KAN*>(factoryForward("KAN", params));
    kan->batchSize = 8;
    kan->summary();
    bool verbosity = true;
    float learningRate = 0.01;
    int epochs = 30;
    int patience = 8;
    trainer(*kan, X_train, Y_train, learningRate, epochs, verbosity, patience);
    saveWeights(*kan, "test_wegihts_saved.json");
    test(*kan, X_test, Y_test);
    kan->loadWeights("test_wegihts_saved.json");
    test(*kan, X_test, Y_test);
    return 0;
}