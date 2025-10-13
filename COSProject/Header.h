#pragma once
#include <Eigen/Dense>


class FeedForward {
public:
	FeedForward();
	virtual Eigen::VectorXd forward(Eigen::VectorXd) = 0;
};

class RBF : public FeedForward
{
public:
	// contructors
	RBF(double, double, int);
	RBF();
	// memeber functions
	virtual Eigen::VectorXd forward(Eigen::VectorXd);

	// data mebers
	double start;
	double end;
	double denom;
	Eigen::VectorXd centers;
	Eigen::MatrixXd dRBF(Eigen::VectorXd);
};

class Layer : public FeedForward
{
public:
	Layer(int, int);
	virtual Eigen::VectorXd forward(Eigen::VectorXd);

	int num;
	int inputDimension;
	Eigen::MatrixXd weights;
};

class KAN : public FeedForward {
public:
	KAN(int grid, float begin, float end, int inputDimension, int OutputDimension, int HiddenDimension, int numOfLayers);
	
	Eigen::VectorXd psi(Eigen::VectorXd);
	virtual Eigen::VectorXd forward(Eigen::VectorXd);
	void backpropagation(Eigen::VectorXd y, float lr);
	int grid;
	int numOfLayers;
	RBF rbf;
	std::vector<Layer> weights;
	std::vector<Eigen::MatrixXd> dWeights;
	std::vector<Eigen::VectorXd> activations;
	std::vector<Eigen::VectorXd> deltas;
};