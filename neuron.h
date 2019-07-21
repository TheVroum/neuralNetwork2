#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <functional>
#include <queue>
#include <deque>
#include <cmath>
#include <cassert>
#include <map>









namespace jo_nn
{




//make the ++ and -- used callbacks to have const parameters so only value and coefficient are altered


///in the constructor of neuron :
///Cycle indicate a parallel cycle. Each cycle number is not
///a unique computation/assertion but n computation/assertion where n is the number of parallel network training


template <size_t dividableBy = 100>
class layerCoordinate
{
public:


    int n;

    inline layerCoordinate(){n = 0;}
    layerCoordinate(const layerCoordinate&o):n(o.n){}
    inline layerCoordinate(double i): n(static_cast<int>(i*dividableBy + 0.1/*implique que l'arrondi n'est par défault que sur les 9/10*/)){}
    inline layerCoordinate(int i): n(i*100){}
    inline bool operator <(const layerCoordinate&o) const {return n < o.n;}
    inline bool operator==(const layerCoordinate&o) const {return n == o.n;}
    inline operator bool(){return n;}
    inline void operator++(int){n += 100;}
    inline void operator++(){n += 100;}
    inline void operator=(const layerCoordinate&o){n = o.n;}
};



typedef std::pair <layerCoordinate, size_t> neuronCoordinate;

template <typename ExtraDataT>
class neuron;






typedef std::function <double(size_t cycle
    , double coeff/*of the link between the two neurons*/
    , double propagatingError
    , double errorIndicator
    , neuronCoordinate nCoordinate)>
neuronCoeffDerivativeCalculatorFunction;




template <typename ExtraDataT>
using computationFunction = std::function <void(const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &ne
, const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &pr
, neuron<ExtraDataT>*
, neuronCoordinate nCoordinate
, double errorIndicator
, size_t cycle)>;




template <typename ExtraDataT>//No traits. Use pointer to use pointer.
///ExtraDataT MUST be trivial
class neuron
{
public://Attributs

    double bias;
    bool droped;


    double forwardValue;
    double backwardValue;

public://Callbacks to be eventually used by other callbacks

    ExtraDataT ExtraData;

    std::function <double(double input)> c_activationFunction, c_activationFunctionDerivative;

    neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator;

    computationFunction<ExtraDataT> c_forwardCompute, c_backwardCompute;//calls c_activationFunction & c_activationFunctionDerivative

    std::function <void(bool backward, neuron* target)> c_normalize;
    //the previous callback now plays the role of giving directly the derivative, to add to the coeff

    //les paramètres les plus facultatifs sont en haut (inverser donc l'ordre pour pouvoir les omettre).


public://Membres

    neuron() = delete;

    neuron(neuron&&) = default;
    neuron operator=(neuron&&) = default;
    neuron operator=(const neuron&) = default;
    neuron(const neuron&) = default;
    ~neuron() = default;

    neuron(std::function <void(bool direction, neuron* target)> c_normalize_p
        , std::function <double(double input)> c_activationFunction_p
        , std::function <double(double input)> c_activationFunctionDerivative_p
        , neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator_p
        , computationFunction<ExtraDataT> forwardCalculator
        , computationFunction<ExtraDataT> backwardCalculator
        , double bias_p
        , bool droped_p, ExtraDataT ExtraData_p);



    inline void normalize(bool backward/*or back ward if = 0*/);
    inline double wrapperCoeffDerivativeCalculator(double currentCoefficient
        , neuronCoordinate nCoordinate
        , double errorIndicator
        , size_t cycle
        );//calls c_coeffDerivativeCalculator


    inline void operator()(bool backPropagating);

    inline void backC(std::vector<std::pair<neuron<ExtraDataT>*, double *>> &next,
    std::vector<std::pair<neuron<ExtraDataT>*, double *>> &previous
    , neuronCoordinate c, double errorIndicator, size_t cycle);

    inline void forC(std::vector <std::pair <neuron<ExtraDataT>*, double*>> &next,
        std::vector <std::pair <neuron<ExtraDataT>*, double*>> &previous
        , neuronCoordinate c, double errorIndicator, size_t cycle);

};








template <typename ExtraDataT>
neuron<ExtraDataT>::neuron(std::function <void(bool direction, neuron* target)> c_normalize_p
    , std::function <double(double input)> c_activationFunction_p
    , std::function <double(double input)> c_activationFunctionDerivative_p
    , neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator_p
    , computationFunction<ExtraDataT> forwardCalculator_p
    , computationFunction<ExtraDataT> backwardCalculator_p
    , double bias_p
    , bool droped_p
    , ExtraDataT ExtraData_p):
bias(bias_p),
droped(droped_p),
forwardValue(0),
backwardValue(0),
ExtraData(ExtraData_p),
c_activationFunction(c_activationFunction_p),
c_activationFunctionDerivative(c_activationFunctionDerivative_p),
c_coeffDerivativeCalculator(c_coeffDerivativeCalculator_p),
c_forwardCompute(forwardCalculator_p),
c_backwardCompute(backwardCalculator_p),
c_normalize(c_normalize_p)
{
}








template <typename ExtraDataT>
inline void neuron<ExtraDataT>::normalize(bool backward/*or forward if = 0*/)
{
    c_normalize(backward, this);
    return;
}





template <typename ExtraDataT>
inline void neuron<ExtraDataT>::operator()(bool backPropagating)
{
    normalize(backPropagating);
}




template <typename ExtraDataT>
inline double neuron<ExtraDataT>::wrapperCoeffDerivativeCalculator(double currentCoefficient
, neuronCoordinate nCoordinate
, double errorIndicator
, size_t cycle
)
{
    return c_coeffDerivativeCalculator(cycle, currentCoefficient
        , backwardValue, errorIndicator, nCoordinate);
}





template <typename ExtraDataT>
inline void neuron<ExtraDataT>::backC(
    std::vector <std::pair <neuron<ExtraDataT>*, double*>> &next,
    std::vector <std::pair <neuron<ExtraDataT>*, double*>> &previous
    , neuronCoordinate c, double errorIndicator, size_t cycle)
{
    c_backwardCompute(next, previous, this
        , c, errorIndicator, cycle);
}



template <typename ExtraDataT>
inline void neuron<ExtraDataT>::forC(
    std::vector <std::pair <neuron<ExtraDataT>*, double*>> &next,
    std::vector <std::pair <neuron<ExtraDataT>*, double*>> &previous
    , neuronCoordinate c, double errorIndicator, size_t cycle)
{
    c_forwardCompute(next, previous, this
        , c, errorIndicator, cycle);
}


}


#endif // NEURON_H
