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

void runTests() {
	float a = 4, b = 1, c = 2, d = 3;
	int e = 3, f = 4;
	testRBF(a, b, c, d);
	testLayer(a, b, c, e, f);
}