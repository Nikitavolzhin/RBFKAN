#include "Header.h"
#include <iostream>
#include <random>

RBF::RBF(double start, double end, int num)
{
	//To do: write expection for num if it is <2

	this->start = start;
	this->end = end;
	this->denom = (end - start) / (num - 1);
	this->centers = Eigen::VectorXd::LinSpaced(num, start, end);
}

RBF::RBF() : RBF(0, 1, 1)
{
}

Eigen::VectorXd RBF::forward(Eigen::VectorXd value) // to do: rewrite to pass by reference 
{
	// broacasting
	// rewrite in a better way
	Eigen::MatrixXd v1 = this->centers.transpose().replicate(value.size(), 1);
	Eigen::MatrixXd v2 = value.replicate(1, this->centers.size());
	Eigen::MatrixXd output = (-((v1 - v2) / this->denom).array().square()).exp().matrix();
	Eigen::Map<const Eigen::VectorXd> flatOutput(output.data(), output.size());


	// gaussian RBF
	return flatOutput;
}

Eigen::MatrixXd RBF::dRBF(Eigen::VectorXd value)
{
	Eigen::MatrixXd v1 = this->centers.transpose().replicate(value.size(), 1);
	Eigen::MatrixXd v2 = value.replicate(1, this->centers.size());
	Eigen::MatrixXd output = (-((v1 - v2) / this->denom).array().square()).exp().matrix();
	Eigen::MatrixXd dRBFMatrix = -2 / (this->denom) * output.cwiseProduct(v1 - v2);
	Eigen::Map < const Eigen::VectorXd > dRBF(dRBFMatrix.data(), dRBFMatrix.size());
	return dRBF;
}

Layer::Layer(int inputDimension, int outputDimesion)
{
	this->num = num;
	this->inputDimension = inputDimension;
	this->weights = Eigen::MatrixXd::Random(outputDimesion, inputDimension);
}

Eigen::VectorXd Layer::forward(Eigen::VectorXd input)
{
	//Eigen::Map<const Eigen::VectorXd> flatInput(input.data(), input.size());

	return this->weights * input;
}

KAN::KAN(int grid, float begin, float end, int inputDimension, int OutputDimension, int HiddenDimension, int numOfLayers)
{
	if (numOfLayers == 1) {
		weights.push_back(Layer(grid*inputDimension, HiddenDimension));
	}
	else
	{
		for (int i = 0; i < numOfLayers; ++i) {
			if (i + 1 == numOfLayers)
				weights.push_back(Layer(grid*HiddenDimension, OutputDimension));
			else if (i == 0)
				weights.push_back(Layer(grid*inputDimension, HiddenDimension));
			else
				weights.push_back(Layer(grid*HiddenDimension, HiddenDimension));
		}
	}

	rbf = RBF(begin, end, grid);
	this->grid = grid;
}

Eigen::VectorXd KAN::forward(Eigen::VectorXd x)
{
	if (activations.empty()) {
		activations.push_back(x);
		for (int i = 0; i < weights.size(); ++i) {
			activations.push_back(weights[i].forward(rbf.forward(activations[i])));
			std::cout << rbf.forward(activations[i]) << "\n\n";
		}
	}
	else {
		activations[0] = x;
		for (int i = 0; i < weights.size(); ++i) {
			activations[i+1] = weights[i].forward(rbf.forward(activations[i]));
		}
	}
	return activations[activations.size()-1];
}

void KAN::backpropagation(Eigen::VectorXd y_hat, Eigen::VectorXd y)
{
	deltas.clear();
	dWeights.clear();
	deltas.push_back(-y+ activations[activations.size() - 1]);
	dWeights.push_back(deltas[0] * rbf.forward(activations[activations.size() - 2]).transpose());
	//deltas.push_back(psi((weights[0].weights.transpose() * deltas[0]).cwiseProduct(rbf.dRBF(activations[activations.size() - 2]))));
	weights[0].weights -= 0.2*Eigen::Map<const Eigen::MatrixXd>(dWeights[0].data(), weights[0].weights.rows(), weights[0].weights.cols()).eval();

}

Eigen::VectorXd KAN::psi(Eigen::VectorXd x)
{
	Eigen::VectorXd output(x.size() / grid);
	
	for (int i = 0; i < x.size() / grid; i++) {
		output(i) = x.segment(grid * i, grid).sum();
	}
	return output;
}
