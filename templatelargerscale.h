#ifndef TEMPLATELARGERSCALE_H
#define TEMPLATELARGERSCALE_H


#include "templateneuronscale.h"







template <typename ExtraDataT>
struct neuronConstructorParameters;









template <typename ExtraDataT>
neuronConstructorParameters<ExtraDataT> defaultRelu(const neuronCoordinate &)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory<ExtraDataT>;
    ret.c_activationFunction_p = relu;
    ret.c_activationFunctionDerivative_p = reluD;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator;
    ret.forwardCalculator = defaultForwardCompute<ExtraDataT>;
    ret.backwardCalculator = defaultBackwardCompute<ExtraDataT>;
    ret.bias_p = 0;
    ret.historySize = 0;
    ret.droped = 0;
    return ret;
}






template <typename ExtraDataT>
neuronConstructorParameters<ExtraDataT> defaultSoftmax(const neuronCoordinate &)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory<ExtraDataT>;
    ret.c_activationFunction_p = relu;
    ret.c_activationFunctionDerivative_p = reluD;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator;
    ret.forwardCalculator = defaultForwardCompute<ExtraDataT>;
    ret.backwardCalculator = defaultBackwardCompute<ExtraDataT>;
    ret.bias_p = 0;
    ret.historySize = 0;
    return ret;
}








typedef std::pair<std::pair<size_t, size_t>, double/*initial weight*/> layerConnections;


template <typename ExtraDataT>
std::vector <layerConnections> defaultDense(std::vector <size_t> feederDim, std::vector<size_t> fedDim)
{
    std::vector <layerConnections> ret;
    for(size_t i = neuralNetwork<ExtraDataT>::totalSize(feederDim) - 1; i + 1; --i)
        for(auto j = neuralNetwork<ExtraDataT>::totalSize(fedDim) - 1; j + 1; --j)
            ret.push_back(std::pair<std::pair<size_t, size_t>, double>(std::pair<size_t, size_t>(i, j), 1));
    return ret;
}








template <typename ExtraDataT>
inline void emptyInterComputationNeuronAlterationFunction
    (neuron<ExtraDataT>*, neuronCoordinate, size_t)
{}
















#endif // TEMPLATELARGERSCALE_H
