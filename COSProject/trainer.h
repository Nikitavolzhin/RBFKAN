#pragma once
#include <Eigen/Dense>
#include "Header.h"

void trainer(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y, float lr, int epochs, bool verbosity);