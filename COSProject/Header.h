#pragma once
#include <Eigen/Dense>

class RBF
{
public:
	// contructors
	RBF(double, double, int);
	RBF();
	// memeber functions
	Eigen::VectorXd forward(Eigen::VectorXd);

//private:
	// data mebers
	double start;
	double end;
	double denom;
	Eigen::VectorXd centers;
	Eigen::MatrixXd dRBF(Eigen::VectorXd);
};

class Layer
{
public:
	Layer(int, int);
	Eigen::VectorXd forward(Eigen::VectorXd);

//private:
	int num;
	int inputDimension;
	Eigen::MatrixXd weights;
};

class KAN {
public:
	KAN(int grid, float begin, float end, int inputDimension, int OutputDimension, int HiddenDimension, int numOfLayers);
	
	Eigen::VectorXd psi(Eigen::VectorXd);
	Eigen::VectorXd forward(Eigen::VectorXd);
	void backpropagation(Eigen::VectorXd y);
	int grid;
	int numOfLayers;
	RBF rbf;
	std::vector<Layer> weights;
	std::vector<Eigen::MatrixXd> dWeights;
	std::vector<Eigen::VectorXd> activations;
	std::vector<Eigen::VectorXd> deltas;
};