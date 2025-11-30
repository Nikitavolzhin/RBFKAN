#include "classes.h"
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cmath>
#include <stdexcept>

#include <fstream>
#include "json.hpp"

RBF::RBF(double start, double end, int gridSize)
{
	if (end <= start) {
		std::cout << "Error: incorrect start/end RBF values. Used default -1/1 instead\n";
		start = -1;
		end = 1;
	}
	if (gridSize < 2) {
		std::cout << "Error: incorrect grid size for RBF. Used default values of 2 instead\n";
		gridSize = 2;
	}

	this->start = start;
	this->end = end;
	this->denom = (end - start) / (gridSize - 1);
	this->centers = Eigen::VectorXd::LinSpaced(gridSize, start, end);

}

RBF::RBF() : RBF(0, 1, 1)
{
}

Eigen::VectorXd RBF::forward(const Eigen::VectorXd& value)  
{

	Eigen::MatrixXd v1 = this->centers.transpose().replicate(value.size(), 1);
	Eigen::MatrixXd v2 = value.replicate(1, this->centers.size());
	Eigen::MatrixXd output = v1 - v2;
	output = output / this->denom;
	output = (-output.array().square()).exp().matrix();
	Eigen::Map<const Eigen::VectorXd> flatOutput(output.data(), output.size());


	return flatOutput;
}

Eigen::MatrixXd RBF::dRBF(Eigen::VectorXd& value)
{
	Eigen::MatrixXd v1 = this->centers.transpose().replicate(value.size(), 1);
	Eigen::MatrixXd v2 = value.replicate(1, this->centers.size());
	Eigen::MatrixXd output = v1 - v2;
	output = output / this->denom;
	output = (-output.array().square()).exp().matrix();
	Eigen::MatrixXd dRBFMatrix = -2 / (this->denom) * output.cwiseProduct(v1 - v2);
	Eigen::Map < const Eigen::VectorXd > dRBF(dRBFMatrix.data(), dRBFMatrix.size());
	return dRBF;
}

Layer::Layer(unsigned inputDimension, unsigned outputDimension, std::string initialization)
{

	Eigen::Rand::Vmt19937_64 gen;

	this->inputDimension = inputDimension;

	if (initialization=="uniform")
		this->weights = Eigen::MatrixXd::Random(outputDimension, inputDimension);
	else if (initialization == "normal") {
		this->weights = Eigen::Rand::normal<Eigen::MatrixXd>(outputDimension, inputDimension, gen, 0.00, 1.0);
	}
	else if (initialization == "He") {
		this->weights = Eigen::Rand::normal<Eigen::MatrixXd>(outputDimension, inputDimension, gen, 0.00, 2.0 / float(inputDimension));
	}
	else if (initialization == "Glorot"){
		this->weights = Eigen::MatrixXd::Random(outputDimension, inputDimension) * sqrt(6.0 / float(outputDimension + inputDimension));
	}
	
	
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input)
{
	if (input.size() != this->weights.cols()) {
		throw std::invalid_argument("Layer dimension mismatch");
	}
	return this->weights * input;
}

KAN::KAN(config params)
{
	this->params = params;
	if (params.numOfLayers == 1) {
		weights.push_back(dynamic_cast<Layer*>(factoryForward("sinlgeLayer", params)));
	}
	else
	{
		for (int i = 0; i < params.numOfLayers; ++i) {
			if (i + 1 == params.numOfLayers)
				weights.push_back(dynamic_cast<Layer*>(factoryForward("outputLayer", params)));
			else if (i == 0)
				weights.push_back(dynamic_cast<Layer*>(factoryForward("inputLayer", params)));
			else
				weights.push_back(dynamic_cast<Layer*>(factoryForward("hiddenLayer", params)));
		}
	}

	rbf = dynamic_cast<RBF*>(factoryForward("RBF", params));
}

Eigen::VectorXd KAN::forward(const Eigen::VectorXd& x)
{

	if (testing) {
		Eigen::VectorXd result = x;
		for (int i = 0; i < weights.size(); ++i) {
			result = (weights[i]->forward(rbf->forward(result)));
		}
		return result;
	}
	if (activations.empty()) {
		activations.push_back(x);
		for (int i = 0; i < weights.size(); ++i) {
			activations.push_back(weights[i]->forward(rbf->forward(activations[i])));
		}
	}
	else {
		activations[0] = x;
		for (int i = 0; i < weights.size(); ++i) {
			activations[i+1] = weights[i]->forward(rbf->forward(activations[i]));
		}
	}
	return activations[activations.size()-1];
}

void KAN::summary()
{
	std::cout << "KAN Summary\n";
	std::cout << "Grid size: " << params.gridSize << "\n";
	std::cout << "Grid start: " << params.start << "\n";
	std::cout << "Grid end: " << params.end << "\n";
	std::cout << "Layers: " << params.numOfLayers << "\n";
	for (int i = 0; i < weights.size(); ++i) {
		std::cout << " Layer " << i
			<< ": " << weights[i]->weights.rows()
			<< " x " << weights[i]->weights.cols() << "\n";
	}
	std::cout << "Batch size: " << batchSize << "\n";
	std::cout << "Mode: ";
	if (testing)
		std::cout << " tesing\n";
	else
		std::cout << " training \n";
}

void KAN::backpropagation(Eigen::VectorXd& y, float lr)
{
	Eigen::VectorXd delta;
	Eigen::VectorXd y_hat = activations[activations.size() - 1];
	if (this->batchSize == iteration) {
		dWeights.clear();
		iteration = 0;
	}
	deltas.clear();
	//first delta and gradient
	if (params.loss == "MAE") {
		Eigen::RowVectorXd signs(y.size());
		for (int i = 0; i < y.size(); ++i) {
			if (y_hat[i] - y[i] < 0)
				signs << -1;
			else
				signs << 1;
		}
		deltas.push_back(signs);
	} 
	else if (params.loss == "MSE")
		deltas.push_back(y_hat - y);
	if (iteration == 0)
		dWeights.push_back(deltas[0] * rbf->forward(activations[activations.size() - 2]).transpose());
	else
		dWeights[0] += deltas[0] * rbf->forward(activations[activations.size() - 2]).transpose();
	if (batchSize==1)
		weights[params.numOfLayers - 1]->weights -= lr * dWeights[0];
	else if (iteration + 1 == batchSize) {
		weights[params.numOfLayers - 1]->weights -= lr * dWeights[0]/ batchSize;
	}
	//all consequtive deltas and graidents
	for(int i = 1; i < params.numOfLayers; ++i) {
		delta = (weights[params.numOfLayers - i]->weights.transpose() * deltas[i - 1]);
		delta = delta.cwiseProduct(rbf->dRBF(activations[activations.size() - 1 - i]));
		delta = psi(delta);
		deltas.push_back(delta);
		if (batchSize == 1){
			dWeights.push_back(deltas[i] * rbf->forward(activations[activations.size() -2-i]).transpose());
			weights[params.numOfLayers - 1 - i]->weights -= lr * dWeights[i];
		}
		else if (iteration == 0) {
			dWeights.push_back(deltas[i] * rbf->forward(activations[activations.size() - 2 - i]).transpose());
		}
		else {
			dWeights[i] += deltas[i] * rbf->forward(activations[activations.size() - 2 - i]).transpose();
			if (iteration + 1 == batchSize) {
				weights[params.numOfLayers - 1 - i]->weights -= lr * dWeights[i]/ batchSize;
			}
		}
	}
	iteration++;

}

Eigen::VectorXd KAN::psi(const Eigen::VectorXd& x)
{
	int newSize = x.size() / params.gridSize;
	Eigen::VectorXd output(newSize);
	
	for (int i = 0; i < newSize; i++) {
		output(i) = x.segment(params.gridSize * i, params.gridSize).sum();
	}
	return output;
}

void KAN::loadWeights(const std::string& path) {
	
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cout << "Error: cannot open the file";
		return;
	}
	nlohmann::json savedWeights;
	file >> savedWeights;
	file.close();
	
	for (int i = 0; i < params.numOfLayers; ++i) {
		std::vector<double> data = savedWeights[i]["entries"];
		if (data.size() != weights[i]->weights.rows()* weights[i]->weights.cols())
		{
			std::cout << "Error: matrix size mismatch in saved weights for layer "
				<< i << "\n";
			return;
		}
		weights[i]->weights = Eigen::Map<Eigen::MatrixXd>(data.data(), int(savedWeights[i]["rows"]), int(savedWeights[i]["cols"]));
		
	}
	std::cout << "Success. Weights loaded\n";

}