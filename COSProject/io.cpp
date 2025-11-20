#include "io.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "json.hpp"


Eigen::MatrixXd readCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Error: Could not open the file";
        return Eigen::MatrixXd::Zero(1, 1);
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

void writeCSV(const Eigen::MatrixXd& matrix, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cout << "Error: Could not open the file";
        return;
    }

    int rows = matrix.rows();
    int cols = matrix.cols();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix(i, j);
            if (j < cols - 1) file << ',';
        }
        file << '\n';
    }
}

config readConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file");
    }
    nlohmann::json j;
    file >> j;
    config params;

    params.start = j["start"];
    params.end = j["end"];
    params.gridSize = j["gridSize"];
    params.inputDimension = j["inputDimension"];
    params.outputDimension = j["outputDimension"];
    params.numOfLayers = j["numOfLayers"];
    params.hiddenDimension = j["hiddenDimension"];
    params.initialization = j["initialization"];
    params.loss = j["loss"];

    return params;
}
