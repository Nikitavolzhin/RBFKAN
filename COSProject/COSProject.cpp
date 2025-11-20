#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Header.h"
#include "io.h"
#include "trainer.h"

int main()
{
    Eigen::MatrixXd mat = readCSV("data_1.csv");
    Eigen::MatrixXd X = mat.leftCols(2);
    Eigen::MatrixXd Y = mat.rightCols(1);
    
    config params = readConfig("config.json");
    
    KAN* kan = dynamic_cast<KAN*>(factoryForward("KAN", params));
    kan->batchSize = 1;
    trainer(*kan, X, Y, true);
    trainer(*kan, X, Y, true);
    return 0;
}