#include <iostream>
#include "Header.h"


void testRBF(float a, float b, float c, float d) {
	Eigen::VectorXd input{ {a, b, c, d} };
	RBF rbf = RBF(-1, 1, 3);
	if (rbf.denom == 1)
		std::cout << "RBF denominator OK" << "\n";
	else {
		std::cout << "Failed due to incorrect RBF denominator\n";
		return;
	}
	if ((rbf.centers[0] == -1) and (rbf.centers[1] == 0) and (rbf.centers[2] == 1))
		std::cout << "RBF centers OK" << "\n";
	else {
		std::cout << "Failed due to incorrect RBF centers\n";
		return;
	}
	
	Eigen::VectorXd result = rbf.forward(input);
	if (result.size() == 12)
		std::cout << "RBF output size OK\n";
	else {
		std::cout << "Failed due to incorrect RBF output shape\n";
		return;
	}
	result = result.cwiseAbs().array().log();
	if ((float(result[0]) == -(-1 - a) * (-1 - a)) and (float(result[5]) == -b*b) and (float(result[11]) == (d - 1) * (1 - d)))
		std::cout << "RBF forward OK \n";
	else {
		std::cout << "Failed due to incorrect RBF computation\n";
		return;
	}	
}

void testLayer(float a, float b, float c, int e, int f) {
	
	Layer layer = Layer(e, f, "Glorot");
	if ((layer.weights.rows() == f) and (layer.weights.cols() == e) and (layer.inputDimension == e))
		std::cout << "Layer matrix shape OK\n";
	else {
		std::cout << "Failed due to incorrect Layer matrix shape\n";
		return;
	}

	
	Eigen::VectorXd input{ {a, b, c} };
	Eigen::VectorXd output = layer.forward(input);
	if (output.size() == f) {
		std::cout << "Layer forward output shape OK\n";
	}
	else {
		std::cout << "Failed due to incorrect forward output shape\n";
		return;
	}
	if (layer.weights * input == output)
		std::cout << "Layer forward OK\n";
	else {
		std::cout << "Failed due to incorrect Layer matrix multiplication\n";
		return;
	}
}

void testKAN(float a, float b, float c, float d, int f, int g, int h) {
	config params;
	params.end = -1;
	params.gridSize = 3;
	params.hiddenDimension = f;
	params.initialization = "Glorot";
	params.inputDimension = g;
	params.loss = "MSE";
	params.numOfLayers = 2;
	params.outputDimension = h;
	params.start = 1;
	KAN kan = KAN(params);
	Eigen::VectorXd input{ {a, b, c, -b, d, c, -c, -a, b} };
	Eigen::VectorXd output = kan.psi(input);
	if ((output[0] == a + b + c) and (output[1] == c - b + d) and (output[2] == b - a - c)) {
		std::cout << "PSI function OK\n";
	}
	else {
		std::cout << "Failure due to incorrect psi function in KAN\n";
		return;
	}
	if ((kan.weights[0]->weights.rows() == f) and (kan.weights[0]->weights.cols() == 3 * g))
		std::cout << "KAN onput layer OK\n";
	else {
		std::cout << "Failed due to incorrect KAN intput layer\n";
		return;
	}
	if ((kan.weights[1]->weights.rows() == h) and (kan.weights[1]->weights.cols() == 3 * f))
		std::cout << "KAN output layer OK\n";
	else {
		std::cout << "Failed due to incorrect KAN output layer\n";
		return;
	}

}

void runTests() {
	float a = 4, b = 1, c = 2, d = 3;
	int e = 3, f = 5, g = 4, h = 2;
	testRBF(a, b, c, d);
	testLayer(a, b, c, e, f);
	testKAN(a, b, c, d, f, g, h);
}