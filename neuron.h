#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <functional>
#include <queue>
#include <deque>
#include <cmath>
#include <cassert>
#include <map>

///This library only support DAG type neural network. For instance, it doesnt support recurrent neural network.
//expand to non dag with my idea
//expand to general history connection


//on next version, dissociate addition of values based on historical to allow ful parallelization of historical value based (pre-)calculus


///les connections sont strictement successives : on ne peut pas prendre la valeur n-2 et pas la valeur n-1
///Be careful to inline the higly used callbacks




//make the ++ and -- used callbacks to have const parameters so only value and coefficient are altered


///in the constructor of neuron :
///Cycle indicate a parallel cycle. Each cycle number is not
///a unique computation/assertion but n computation/assertion where n is the number of parallel network training




///be careful. Copy operator is partial




class layerCoordinate
{
public:


    int n;

    inline layerCoordinate(){n = 0;}
    layerCoordinate(const layerCoordinate&o):n(o.n){}
    inline layerCoordinate(double i): n(i*10){}
    inline layerCoordinate(int i): n(i*10){}
    inline bool operator <(const layerCoordinate&o) const {return n < o.n;}
    inline bool operator==(const layerCoordinate&o) const {return n == o.n;}
    inline operator bool(){return n;}
    inline void operator++(int){n += 10;}
    inline void operator++(){n += 10;}
    inline void operator=(const layerCoordinate&o){n = o.n;}
};






template <typename ExtraDataT/*on neurons*/>
class neuralNetwork;





class layerCoordinate;
typedef std::pair <layerCoordinate, size_t> neuronCoordinate;




typedef std::function <double(size_t cycle
    , double coeff/*of the link between the two neurons*/
    , double propagatingError
    , double errorIndicator
    , neuronCoordinate nCoordinate)>
neuronCoeffDerivativeCalculatorFunction;



template <typename ExtraDataT>
class neuron;


template <typename ExtraDataT>
using computationFunction = std::function <void(const std::vector <std::pair<size_t, double*>> &sc
                            , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &ne
                            , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &pr
                            , neuron<ExtraDataT>*)>;




template <typename ExtraDataT>//No traits. Use pointer to use pointer.
///ExtraDataT MUST be trivial
class neuron
{
public://Attributs

    double bias;
    bool droped;

    size_t *cycle;
    double *errorIndicator;
    bool *backPropagating;
    neuronCoordinate nCoordinate;

    double forwardValue;
    std::deque <double> forwardValueHistory;
    double backwardValue;
    std::deque <double> backwardValueHistory;

#ifndef NDEBUG
    bool inited;
    bool linked;
#endif




private :

    std::vector <std::pair<size_t, double*>> selfCoeffs;
    std::vector <std::pair <neuron*, double*>> next, previous;
    friend class neuralNetwork<ExtraDataT>;

public:

    ExtraDataT ExtraData;



public://Callbacks to be eventually used by other callbacks


    std::function <double(double input)> c_activationFunction, c_activationFunctionDerivative;


private:


    neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator;

    computationFunction<ExtraDataT> c_forwardCompute, c_backwardCompute;//calls c_activationFunction & c_activationFunctionDerivative

    std::function <void(bool backward, neuron* target)> c_normalize;
    //the previous callback now plays the role of giving directly the derivative, to add to the coeff

    //les paramètres les plus facultatifs sont en haut (inverser donc l'ordre pour pouvoir les omettre).


public://Membres

    neuron() = delete;
    neuron(neuron&&) = delete;
    neuron operator=(neuron&&) = delete;
    neuron operator=(const neuron&) = delete;

    neuron(const neuron&);//delete;//default copy do not handle
    ~neuron() = default;

    neuron(std::function <void(bool direction, neuron* target)> c_normalize_p
        , std::function <double(double input)> c_activationFunction_p
        , std::function <double(double input)> c_activationFunctionDerivative_p
        , neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator_p
        , computationFunction<ExtraDataT> forwardCalculator
        , computationFunction<ExtraDataT> backwardCalculator
        , double bias_p
        , bool droped_p, size_t historySize
        , size_t *cycle_p///Cycle indicate a parallel cycle. Each cycle number is not
                        ///a unique computation/assertion but n computation/assertion where n is the number of parallel network training
        , double *errorIndicator_p
        , bool *backPropagating_p
        , neuronCoordinate nCoordinate_p
        , ExtraDataT ExtraData_p);




    void set/*operator=*/(std::function <void(bool direction, neuron* target)> c_normalize_p
        , std::function <double(double input)> c_activationFunction_p
        , std::function <double(double input)> c_activationFunctionDerivative_p
        , neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator_p
        , computationFunction<ExtraDataT> forwardCalculator
        , computationFunction<ExtraDataT> backwardCalculator
        , double bias_p
        , bool initialDropState_p
        , size_t historySize
        , size_t *cycle_p
        , double *errorIndicator_p
        , bool *backPropagating_p
        , neuronCoordinate nCoordinate_p
        , ExtraDataT ExtraData_p);



    inline void normalize(bool backward/*or back ward if = 0*/);
    inline double wrapperCoeffDerivativeCalculator(
        double currentCoefficient);//calls c_coeffDerivativeCalculator


    inline void operator()();//calls c_normalize though normalize
    inline void operator++();//calls c_forwardCompute
    inline void operator++(int){this->operator++();}//calls c_forwardCompute
    inline void operator--();//calls c_backwardCompute


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
    , size_t historySize
    , size_t *cycle_p
    , double *errorIndicator_p
    , bool *backPropagating_p
    , neuronCoordinate nCoordinate_p
    , ExtraDataT ExtraData_p):
bias(bias_p),
droped(droped_p),
cycle(cycle_p),
errorIndicator(errorIndicator_p),
backPropagating(backPropagating_p),
nCoordinate(nCoordinate_p),
forwardValue(0),
forwardValueHistory(historySize, NAN),
backwardValue(0),
backwardValueHistory(historySize, NAN),
inited(1),
linked(0),
ExtraData(ExtraData_p),
c_activationFunction(c_activationFunction_p),
c_activationFunctionDerivative(c_activationFunctionDerivative_p),
c_coeffDerivativeCalculator(c_coeffDerivativeCalculator_p),
c_forwardCompute(forwardCalculator_p),
c_backwardCompute(backwardCalculator_p),
c_normalize(c_normalize_p)
{
    assert(std::isnan(NAN));//std::numeric_limits::quiet_NaN
}




template <typename ExtraDataT>
void neuron<ExtraDataT>::set(std::function <void(bool direction, neuron* target)> c_normalize_p
    , std::function <double(double input)> c_activationFunction_p
    , std::function <double(double input)> c_activationFunctionDerivative_p
    , neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator_p
    , computationFunction<ExtraDataT> forwardCalculator_p
    , computationFunction<ExtraDataT> backwardCalculator_p
    , double bias_p
    , bool initialDropState_p
    , size_t historySize
    , size_t *cycle_p
    , double *errorIndicator_p
    , bool *backPropagating_p
    , neuronCoordinate nCoordinate_p
    , ExtraDataT ExtraData_p)
{
    bias = bias_p;
    droped = initialDropState_p;
    cycle = cycle_p;
    errorIndicator = errorIndicator_p;
    backPropagating = backPropagating_p;
    nCoordinate = nCoordinate_p;
    forwardValue = 0;
    forwardValueHistory = historySize, NAN;
    backwardValue = 0;
    backwardValueHistory = historySize, NAN;
    ExtraData = ExtraData_p;
    c_normalize = c_normalize_p;
    c_activationFunction = c_activationFunction_p;
    c_activationFunctionDerivative = c_activationFunctionDerivative_p;
    c_coeffDerivativeCalculator = c_coeffDerivativeCalculator_p;
    c_forwardCompute = forwardCalculator_p;
    c_backwardCompute = backwardCalculator_p;
}






template <typename ExtraDataT>
inline void neuron<ExtraDataT>::normalize(bool backward/*or forward if = 0*/)
{
    c_normalize(backward, this);
    return;
}





template <typename ExtraDataT>
inline void neuron<ExtraDataT>::operator()()
{
    normalize(*backPropagating);
}




template <typename ExtraDataT>
inline double neuron<ExtraDataT>::wrapperCoeffDerivativeCalculator(
    double currentCoefficient)
{
    return c_coeffDerivativeCalculator(*cycle, currentCoefficient
        , backwardValue, *errorIndicator, nCoordinate);
}





template <typename ExtraDataT>
inline void neuron<ExtraDataT>::operator--()
{
    c_backwardCompute(selfCoeffs, next, previous, this);
}




template <typename ExtraDataT>
inline void neuron<ExtraDataT>::operator++()
{
    c_forwardCompute(selfCoeffs, next, previous, this);
}





template <typename ExtraDataT>
neuron<ExtraDataT>::neuron(const neuron&n):
    bias(n.bias),
    droped(0),//i chose that droping state is not copied
    cycle(n.cycle),
    errorIndicator(n.errorIndicator),
    backPropagating(n.backPropagating),
    nCoordinate(n.nCoordinate),
    forwardValue(n.forwardValue),
    forwardValueHistory(n.forwardValueHistory),
    backwardValue(n.backwardValue),
    backwardValueHistory(n.backwardValueHistory),
    inited(n.inited),
    linked(n.linked),
    ExtraData(n.ExtraData),
    c_activationFunction(n.c_activationFunction),
    c_activationFunctionDerivative(n.c_activationFunctionDerivative),
    c_coeffDerivativeCalculator(n.c_coeffDerivativeCalculator),
    c_forwardCompute(n.c_forwardCompute),
    c_backwardCompute(n.c_backwardCompute),
    c_normalize(n.c_normalize)
{
    next.clear();
    previous.clear();
    selfCoeffs.clear();
}







#endif // NEURON_H
