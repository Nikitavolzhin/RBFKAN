#pragma once
#include <Eigen/Dense>
#include <string>
#include "config.h"

Eigen::MatrixXd readCSV(const std::string& path);
void writeCSV(const Eigen::MatrixXd& matrix, const std::string& path);
config readConfig(const std::string& path);