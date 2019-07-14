#ifndef TEMPLATES_H
#define TEMPLATES_H
#include <cmath>
#include <time.h>
#include <cstdlib>
#include <cstdio>


#include "randomness.h"
#include "callbackstemplate.h"
#include "neuron.h"












namespace jo_nn
{

typedef std::pair<std::pair<size_t, size_t>, double/*initial weight*/> layerConnections;

///Be careful to inline the higly used callbacks


///Treatment of droped neuron :
///i chose here, for performance purposes, that instead of checking "to-add-neurons" droping state before
///adding their coefficient, droped neuron coeffient will be set to 0. However depending on the calculation
///function used, this could be a problem in the case where 0 is not a neutral element (like if it was
///a multiplicative calculus).



///create here a normalizeDropout callback











inline double relu(double input)
{
    if(input < 0)
        return 0;
    else
        return input;
}

inline double reluD(double input)
{
    if(input < 0)
        return 0;
    else
        return 1;
}


template <size_t leakyCoeff/*millieme*/>
inline double lkRelu(double input)
{
    if(input < 0)
        return input * leakyCoeff / 1000;
    else
        return input;
}

template <size_t leakyCoeff/*millieme*/>
inline double lkReluD(double input)
{
    if(input < 0)
        return leakyCoeff / 1000;
    else
        return 1;
}





template <size_t learningRate/*millionieme*/>
inline double defaultcoeffDerivativeCalculator(size_t/* cycle*//*parameter name commented to prevent warnings, uncomment for use with your own callback*/
    , double coeff
    , double propagatingError, double /*errorIndicator*/
    , neuronCoordinate /*nCoordinate*/)
{
    if(/*signbit(*/propagatingError * coeff/*)*/ < 0.0)
        return propagatingError * (learningRate / 1000000.0);
    if(std::abs(coeff) < 10.0)
        return propagatingError * (learningRate / 1000000.0);
    else
        return propagatingError / (coeff * (100000.0 / learningRate));
}

template <typename ExtraDataT>
void normalizeNoHistory(bool backward, neuron<ExtraDataT>* target)//bias added in forward compute
{
    if(!backward)
        target->forwardValue = 0;
    else
        target->backwardValue = 0;
    return;
}


template <typename ExtraDataT>
void defaultForwardCompute(const std::vector <std::pair<size_t, double*>> &//sc//no self recurrency
//bias added here
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &pr
    , neuron<ExtraDataT>* n
    , neuronCoordinate
    , double
    , size_t)
{
    if(!n->droped)
    {
        /*for(auto a : sc)
            n->forwardValue += (*(a.second)) * n->forwardValueHistory[a.first];*/
        for(auto a : pr)
            n->forwardValue += (*(a.second)) * a.first->forwardValue;
        n->forwardValue += n->bias;
        n->forwardValue = n->c_activationFunction(n->forwardValue);
    }
}



template <typename ExtraDataT>///recurrent coefficient not handled here ! Add them and optionally, add a second wrapperCoeffDerivator callback to calculate these specific coefficients
void defaultBackwardCompute(const std::vector <std::pair<size_t, double*>> &//no self recurrency
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &ne
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &pr
    , neuron<ExtraDataT>* n
    , neuronCoordinate nCoordinate
    , double errorIndicator
    , size_t cycle)
{
    if(!n->droped)
    {
        //on ne peut pas récupérer les dérivées inverse du futur (cependant on pourrait dans le futur compenser)
        for(auto &a : ne)
            n->backwardValue += (*(a.second)) * a.first->backwardValue;
        n->backwardValue *= n->c_activationFunctionDerivative(n->forwardValue);
        for(auto &a : pr)
            (*(a.second)) += n->wrapperCoeffDerivativeCalculator(*(a.second), nCoordinate, errorIndicator, cycle);//automatically gets n->backwardValue
    }
}


template <typename ExtraDataT>
inline void emptyInterComputationNeuronAlterationFunction
    (neuron<ExtraDataT>*, neuronCoordinate, double, size_t)
{}

template <typename ExtraDataT>
std::vector <layerConnections> defaultDense(std::vector <size_t> feederDim, std::vector<size_t> fedDim)
{
    normalness n(rand);
    std::vector <layerConnections> ret;
    for(size_t i = neuralNetwork<ExtraDataT>::totalSize(feederDim) - 1; i + 1; --i)
        for(auto j = neuralNetwork<ExtraDataT>::totalSize(fedDim) - 1; j + 1; --j)
            ret.push_back(std::pair<std::pair<size_t, size_t>, double>(std::pair<size_t, size_t>(i, j), n() * 5));
    return ret;
}







/*

template <typename ExtraDataT>
neuronConstructorParameters<ExtraDataT> defaultRelu(const neuronCoordinate &, const std::vector<size_t>*)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory<ExtraDataT>;
    ret.c_activationFunction_p = relu;
    ret.c_activationFunctionDerivative_p = reluD;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator;
    ret.forwardCalculator = defaultForwardCompute<ExtraDataT>;
    ret.backwardCalculator = defaultBackwardCompute<ExtraDataT>;
    ret.bias_p = -0.1;
    ret.historySize = 0;
    ret.droped = 0;
    return ret;
}
*/



template <typename ExtraDataT, int bias = 0/*millionieme*/, size_t learningRate/*millionieme*/ = 10000>
neuronConstructorParameters<ExtraDataT> defaultRelu(const neuronCoordinate, const std::vector<size_t>*)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory<ExtraDataT>;
    ret.c_activationFunction_p = relu;
    ret.c_activationFunctionDerivative_p = reluD;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator<learningRate>;//<100>;
    ret.forwardCalculator = defaultForwardCompute<ExtraDataT>;
    ret.backwardCalculator = defaultBackwardCompute<ExtraDataT>;
    ret.bias_p = static_cast<double>(bias) / 1000000;
    ret.historySize = 0;
    ret.droped = 0;
    return ret;
}

template <typename ExtraDataT, int bias = 0/*millionieme*/, size_t leakyCoeff = 200/*millième*/, size_t learningRate/*millionieme*/ = 10000>
neuronConstructorParameters<ExtraDataT> defaultLkRelu(const neuronCoordinate, const std::vector<size_t>*)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory<ExtraDataT>;
    ret.c_activationFunction_p = lkRelu<leakyCoeff>;
    ret.c_activationFunctionDerivative_p = lkReluD<leakyCoeff>;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator<learningRate>;//<100>;
    ret.forwardCalculator = defaultForwardCompute<ExtraDataT>;
    ret.backwardCalculator = defaultBackwardCompute<ExtraDataT>;
    ret.bias_p = static_cast<double>(bias) / 1000000;
    ret.historySize = 0;
    ret.droped = 0;
    return ret;
}















//Use 8 as a
template <int a = 8>
inline double myOutput(double input)
{
    input = (1 + (a / ((input * input) + 1))) * exp(-input);
    return 1 / (input + 1);
}



template <int a = 8>
inline double myOutputD(double input)
{
    double ei = exp(input), i2 = 2 * input, iip1 = input * input + 1, denom = ((iip1 + a) / ei) + iip1,
        c = i2 / denom, d = (iip1 / (denom * denom)) * (((iip1 + a) / ei) - (i2 / ei) - (i2));
        return c + d;
}






template <typename ExtraDataT>
neuronConstructorParameters<ExtraDataT> defaultSoftmax(const neuronCoordinate, const std::vector<size_t>*)
{
    neuronConstructorParameters<ExtraDataT> ret;
    ret.c_normalize_p = normalizeNoHistory<ExtraDataT>;
    ret.c_activationFunction_p = myOutput;
    ret.c_activationFunctionDerivative_p = myOutputD;
    ret.c_coeffDerivativeCalculator_p = defaultcoeffDerivativeCalculator;
    ret.forwardCalculator = defaultForwardCompute<ExtraDataT>;
    ret.backwardCalculator = defaultBackwardCompute<ExtraDataT>;
    ret.bias_p = 0;
    ret.historySize = 0;
    return ret;
}



//for automation, typically create a first callback that also takethe dimension of layers and
//where learning should decrese, increase, etc.. and use std::bind on it to create specialized version



template <typename ExtraDataT>
void defaultNormalize(bool backward, neuron<ExtraDataT>* target)//set backward to true AFTER normalization
//bias added in forward compute
{
    if(!backward)
        target->forwardValueHistory.push_front(target->forwardValue)
        , target->forwardValueHistory.pop_back()
        , target->forwardValue = 0;
    else
        target->backwardValueHistory.push_front(target->backwardValue)
            , target->backwardValueHistory.pop_back()
            , target->backwardValue = 0;
}












}

















#endif // TEMPLATES_H
