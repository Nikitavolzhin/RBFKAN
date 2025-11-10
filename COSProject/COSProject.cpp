#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include <cstdlib>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Header.h"
#include "config.h"

Eigen::MatrixXd readCSV(std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Error: Could not open the file";
        return Eigen::MatrixXd::Zero(1,1);
    }
    std::string line;
    std::string value;
    std::vector<std::vector<double>> values;
    double number;
    int lineNum = 0;
    while (std::getline(file, line)) {
        values.push_back(std::vector<double>());
        for (char i : line) {
            if ((i != ',') and (i != ';')) {
                value += i;
            }
            else {
                number = std::stod(value); 
                values[lineNum].push_back(number);
                value = "";
            }
        }
        
        number = std::stod(value);
        values[lineNum].push_back(number);
        lineNum++;
        value = "";
    }
    file.close();

    int cols = values[0].size();
    int rows = values.size();
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        mat.row(i) = Eigen::VectorXd::Map(values[i].data(), values[i].size());
    }
    return mat;
}

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
    Eigen::MatrixXd mat = readCSV("data_1.csv");
    Eigen::MatrixXd X = mat.leftCols(2);
    Eigen::MatrixXd Y = mat.rightCols(1);
    
    config params;
    params.start = -2;
    params.end = 2;
    params.gridSize = 5;
    params.inputDimension = 2;
    params.outputDimension = 1;
    params.numOfLayers = 2;
    params.hiddenDimension = 3;
    params.initialization = "He";
    params.loss = "MAE";

    
    KAN* kan = dynamic_cast<KAN*>(factoryForward("KAN", params));
    kan->batchSize = 1;
    //Eigen::MatrixXd X = Eigen::MatrixXd::Random(1000, 1);
    //Eigen::MatrixXd Y = X.array().square();
    trainer(*kan, X, Y);
    trainer(*kan, X, Y);
    return 0;
}