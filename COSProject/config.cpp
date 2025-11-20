#include "config.h"
#include "Header.h"

FeedForward* factoryForward(std::string networkType, config params) {

	if (networkType == "RBF") {
		return new RBF(params.start, params.end, params.gridSize);
	}
	else if (networkType == "sinlgeLayer") {
		return new Layer(params.gridSize * params.inputDimension, params.outputDimension, params.initialization);
	}
	else if (networkType == "hiddenLayer") {
		return new Layer(params.gridSize * params.hiddenDimension, params.hiddenDimension, params.initialization);
	}
	else if (networkType == "inputLayer") {
		return new Layer(params.gridSize * params.inputDimension, params.hiddenDimension, params.initialization);
	}
	else if (networkType == "outputLayer") {
		return new Layer(params.gridSize * params.hiddenDimension, params.outputDimension, params.initialization);
	}
	else if (networkType == "KAN") {
		return new KAN(
			params
		);
	}
}