#pragma once
#include <string>
struct config {
    double start;
    double end;
    int inputDimension;
    int outputDimension;
    int gridSize;
    int hiddenDimension;
    int numOfLayers;
    std::string initialization;
};
class FeedForward;

FeedForward* factoryForward(std::string networkType, config params);