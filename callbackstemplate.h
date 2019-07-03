#ifndef CALLBACKSTEMPLATE_H
#define CALLBACKSTEMPLATE_H
#include <functional>
#include <vector>







namespace jo_nn
{


template <typename ExtraDataT>
struct neuronConstructorParameters;




typedef std::pair <layerCoordinate, size_t> neuronCoordinate;



typedef std::pair<std::pair<size_t, size_t>, double/*initial weight*/> layerConnections;
















///Callback type 1 and 2 :
/// Activation function, and her derivative.
/// Define the activation function that will be used.
/// This callback might only be called by the users callback
/// who are given neuron ref/ptr
typedef std::function <double(double input)> activationFunctionAndDerivative;


///Callback type 3 and 4 :
/// Forward calculation and backward calculation
/// This function is called on every neuron (node) to calculate
/// its output product/its back derivative
/// These call...[detail when theyare called]
template <typename ExtraDataT>
using computationFunction = std::function <void(const std::vector <std::pair<size_t, double*>> &sc
                            , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &ne
                            , const std::vector <std::pair <neuron<ExtraDataT>*, double*>> &pr
                            , neuron<ExtraDataT>*)>;


///Callback type 5
/// Coefficient change culator
///
///
///
typedef std::function <double(size_t cycle
    , double coeff/*of the link between the two neurons*/
    , double propagatingError
    , double errorIndicator
    , neuronCoordinate nCoordinate)>
neuronCoeffDerivativeCalculatorFunction;



///Callback type 6
/// Neuron normalization callback
///
///
///This callback is called after each forward or back computaion on each neuron and the first parameter is the direction that was just calculated.
template <typename ExtraDataT>
using neuronNormalizationCallback = std::function <void(bool backward, neuron<ExtraDataT>* target)>;




///Callback type 7
/// Inter Computation Neuron Alteration Function
///
///
///
template <typename ExtraDataT>
using interComputationNeuronAlterationFunction = std::function<void(neuron<ExtraDataT>* target, neuronCoordinate c, size_t cycle)>;




///Callback type 8
/// Neuron Configure Function (and reconfiguration)
///Is called once on each neuron
///
///
template <typename ExtraDataT>
using neuronConfigureFunction = std::function <neuronConstructorParameters<ExtraDataT>(const neuronCoordinate n, const std::vector<size_t>*dimensionsOfTheLayer)>;



///Callback type 9
/// Layers Connect Function
/// gives the connections to make and their weight.
///
///
typedef std::function <std::vector<layerConnections>(std::vector <size_t>, std::vector<size_t>)> layersConnectFunction;

}







#endif // CALLBACKSTEMPLATE_H
