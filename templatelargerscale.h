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
















#endif // TEMPLATELARGERSCALE_H
