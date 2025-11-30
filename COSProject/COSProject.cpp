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
    
    Eigen::MatrixXd mat = readCSV("sin.csv");
    config params = readConfig("config3.json");
    Eigen::MatrixXd X = mat.leftCols(params.inputDimension);
    Eigen::MatrixXd Y = mat.rightCols(params.outputDimension);
    Eigen::MatrixXd X_train, X_test, Y_train, Y_test;

    trainTestSplit(X, Y, 0.2,   
        X_train, X_test,
        Y_train, Y_test);
    
    
    KAN* kan = dynamic_cast<KAN*>(factoryForward("KAN", params));
    
    kan->batchSize = 16;
    bool verbosity = true;
    float learningRate = 0.02;
    int epochs = 20;
    int patience = 50;
    trainer(*kan, X_train, Y_train, learningRate, epochs, verbosity, patience);
    //saveWeights(*kan, "test_wegihts_saved.json");
    //test(*kan, X_test, Y_test);
    //kan->loadWeights("test_wegihts_saved.json");
    //kan->summary();
    test(*kan, X_test, Y_test);
    
    return 0;
}