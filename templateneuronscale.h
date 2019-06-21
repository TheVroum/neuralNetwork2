#ifndef TEMPLATENEURONSCALE_H
#define TEMPLATENEURONSCALE_H
#include <cmath>





#include "neuron.h"


///Be careful to inline the higly used callbacks




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





//Use 8 as a
template <int a>
inline double myOutput(double input)
{
    input = (1 + (a / ((input * input) + 1))) * exp(-input);
    return 1 / (input + 1);
}



template <int a>
inline double myOutputD(double input)
{
    double ei = exp(input), i2 = 2 * input, iip1 = input * input + 1, denom = ((iip1 + a) / ei) + iip1,
        c = i2 / denom, d = (iip1 / (denom * denom)) * (((iip1 + a) / ei) - (i2 / ei) - (i2));
        return c + d;
}



//for automation, typically create a first callback that also takethe dimension of layers and
//where learning should decrese, increase, etc.. and use std::bind on it to create specialized version
inline bool defaultcoeffDerivativeCalculator(size_t/* cycle*//*parameter name commented to prevent warnings, uncomment for use with your own callback*/
    , double coeff
    , double propagatingError, double /*errorIndicator*/
    , neuronCoordinate /*nCoordinate*/)
{
    if(/*signbit(*/propagatingError * coeff/*)*/ < 0)
        return propagatingError;
    if(abs(coeff) < 10)
        return propagatingError;
    else
        return propagatingError * 10 / coeff;
}

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
void defaultForwardCompute(const std::vector <std::pair<size_t, double*>> &sc//self recurr included
//bias added here
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &ne
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &pr
    , neuron<ExtraDataT>* n)
{
    for(auto a : sc)
        n->forwardValue += (*(a.second)) * n->forwardValueHistory[a.first];
    for(auto a : pr)
        n->forwardValue += a.second * a.first->forwardValue;
    n->forwardValue += n->bias;
    n->forwardValue = n->c_activationFunction(n->forwardValue);
}



template <typename ExtraDataT>
void defaultBackwardCompute(const std::vector <std::pair<size_t, double*>> &sc//self recurr included
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &ne
    , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &pr
    , neuron<ExtraDataT>* n)
{
    //on ne peut pas récupérer les dérivées inverse du futur (cependant on pourrait dans le futur compenser)
    for(auto &a : ne)
        n->backwardValue += a.second * a.first->backwardValue;
    n->backwardValue *= n->c_activationFunctionDerivative(n->forwardValue);
    for(auto &a : pr)
        (*(a.second)) += n->wrapperCoeffDerivativeCalculator(*(a.second));//automatically gets n->backwardValue
}














#endif // TEMPLATENEURONSCALE_H
