#pragma once
#include <Eigen/Dense>
#include "Header.h"

void trainer(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y, float lr, int epochs, bool verbosity, int patience);
void test(KAN& kan, Eigen::MatrixXd& X, Eigen::MatrixXd& Y);
void trainTestSplit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double testRatio,
    Eigen::MatrixXd& X_train, Eigen::MatrixXd& X_test, Eigen::MatrixXd& Y_train, Eigen::MatrixXd& Y_test);