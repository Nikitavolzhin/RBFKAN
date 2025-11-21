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
    file.close();

}

config readConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file");
    }
    nlohmann::json j;
    file >> j;
    file.close();

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

void writeConfig(config params, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file");
    }
    nlohmann::json j;

    j["start"] = params.start;
    j["end"] = params.end;
    j["gridSize"] = params.gridSize;
    j["inputDimension"] = params.inputDimension;
    j["outputDimension"] = params.outputDimension;
    j["numOfLayers"] = params.numOfLayers;
    j["hiddenDimension"] = params.hiddenDimension;
    j["initialization"] = params.initialization;
    j["loss"] = params.loss;
    file << j.dump(4);
    file.close();
}

void saveWeights(const KAN& kan, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cout << "Error: Could not open the file";
        return;
    }
    nlohmann::json wegihts;
    int layers = kan.params.numOfLayers;

    for (int i = 0; i < layers; ++i) {
        wegihts[i]["rows"] = kan.weights[i]->weights.rows();
        wegihts[i]["cols"] = kan.weights[i]->weights.cols();
        wegihts[i]["entries"] = std::vector<double>(
            kan.weights[i]->weights.data(),
            kan.weights[i]->weights.data() + kan.weights[i]->weights.size()
        );
    }
    file << wegihts.dump(4);
    file.close();
}
