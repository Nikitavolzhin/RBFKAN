#pragma once
#include <Eigen/Dense>
#include "config.h"

class FeedForward {
public:
	virtual ~FeedForward() {}
	FeedForward() = default;
	virtual Eigen::VectorXd forward(const Eigen::VectorXd&) = 0;
};

class RBF : public FeedForward
{
public:
	// contructors
	RBF(double, double, int);
	RBF();
	// memeber functions
	Eigen::VectorXd forward(const Eigen::VectorXd&) override;

	// data mebers
	double start;
	double end;
	double denom;
	Eigen::VectorXd centers;
	Eigen::MatrixXd dRBF(Eigen::VectorXd&);
};

class Layer : public FeedForward
{
public:
	Layer(int, int, std::string);
	virtual Eigen::VectorXd forward(const Eigen::VectorXd&);

	int inputDimension;
	Eigen::MatrixXd weights;
};

class KAN : public FeedForward {
public:
	KAN(config params);
	
	Eigen::VectorXd psi(const Eigen::VectorXd&);
	virtual Eigen::VectorXd forward(const Eigen::VectorXd&);
	void backpropagation(Eigen::VectorXd& y, float lr);
	config params;
	
	RBF* rbf;
	int batchSize = 1;
	int iteration = 0;
	std::vector<Layer*> weights;
	std::vector<Eigen::MatrixXd> dWeights;
	std::vector<Eigen::MatrixXd> weightUpdate;
	std::vector<Eigen::VectorXd> activations;
	std::vector<Eigen::VectorXd> deltas;
	bool testing = false;
};