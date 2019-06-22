#ifndef TEMPLATELARGERSCALE_H
#define TEMPLATELARGERSCALE_H


#include "neuralnetwork.h"
#include "templateneuronscale.h"


template <typename ExtraDataT>
neuronConstructorParameters<ExtraDataT> defaultRelu(const neuronCoordinate &n)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory;
    ret.c_activationFunction_p = relu;
    ret.c_activationFunctionDerivative_p = reluD;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator;
    ret.forwardCalculator = defaultForwardCompute;
    ret.backwardCalculator = defaultBackwardCompute;
    ret.bias_p = 0;
    ret.historySize = 0;
    ret.selfCoeffsNumber = 0;
}






template <typename ExtraDataT>
neuronConstructorParameters<ExtraDataT> defaultSoftmax(const neuronCoordinate &n)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory;
    ret.c_activationFunction_p = relu;
    ret.c_activationFunctionDerivative_p = reluD;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator;
    ret.forwardCalculator = defaultForwardCompute;
    ret.backwardCalculator = defaultBackwardCompute;
    ret.bias_p = 0;
    ret.historySize = 0;
    ret.selfCoeffsNumber = 0;
}
















#endif // TEMPLATELARGERSCALE_H
