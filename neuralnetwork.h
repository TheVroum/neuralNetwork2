#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

///This library only support DAG type neural network. For instance, it doesnt support recurrent neural network.




#include <map>
#include <vector>
#include <mutex>
#include <cassert>
#include <cmath>
#include <initializer_list>




#include "templateneuronscale.h"
#include "neuron.h"



template <typename ExtraDataT>
struct neuronConstructorParameters
{
    std::function <void(bool direction, neuron* target)> c_normalize_p;
    std::function <double(double input)> c_activationFunction_p;
    std::function <double(double input)> c_activationFunctionDerivative_p;
    neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator_p;
    computationFunction forwardCalculator;
    computationFunction backwardCalculator;
    double bias_p;
    size_t historySize;
    size_t selfCoeffsNumber;
    ExtraDataT ExtraData_p;
};



typedef std::pair <size_t, size_t> layerCoordinate;
typedef std::pair <layerCoordinate, size_t> neuronCoordinate;

typedef std::function <neuronConstructorParameters(const neuronCoordinate &n)>
    neuronConfigureFunction;


template <typename ExtraDataT>
class neuralNetwork
{
private://Variables organisatrices à partir desquelles on génère le reste.
    class layerCoordinateCmp
    {
        inline bool operator()(layerCoordinateCmp const&a, layerCoordinateCmp const&b)
        {return (static_cast<double>(a.first) /
                 static_cast<double>(a.second) -
                 static_cast<double>(b.first) /
                 static_cast<double>(b.second)) > 0.00001;}
    };

    class neuronCoordinateCmp
    {
        inline bool operator()(neuronCoordinateCmp const&a, neuronCoordinateCmp const&b)
        {
            if(a.first < a.first)
                return 1;
            else
            {
                if(a.first.first == b.first.first && a.first.second == b.first.second)
                {
                    return a.second < b.second;
                }
                else
                    return 0;

            }
        }
        static inline bool compLayer(neuronCoordinateCmp const&a, neuronCoordinateCmp const&b)
        {
            if(a.first < a.first)
                return 1;
            else
                return 0;
        }
    };



    std::map <layerCoordinate, std::vector <neuron<ExtraDataT>>, layerCoordinateCmp>
        neurons;

    std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp>
        links;
    std::map <neuronCoordinate, std::map <size_t, double>, neuronCoordinateCmp>
        selfLinks;

    std::map <layerCoordinate, std::vector <size_t>, layerCoordinateCmp>
        dimensions;
    std::map <layerCoordinate, std::pair<bool, bool>, layerCoordinateCmp>
        layersInformationInputOutput;


    std::mutex computing;

public:



    void createLink(neuronCoordinate &c1, neuronCoordinate &c2, double initialWeight);


    void createRecursiveLink(neuronCoordinate &c1, size_t recursion_level/*0 = previous*/, double initialWeight);


    neuralNetwork();
    //void build();
    void refresh();



    void addLayer(std::vector <size_t>, layerCoordinate lc = {0, 0}, neuronConfigureFunction);
    void addLayer(std::initializer_list <size_t>, layerCoordinate lc = {0, 0}, neuronConfigureFunction);

    void alterLayer(layerCoordinate lc, neuronConfigureFunction);


    void connect()






private://computation time Attribute

    double errorIndicator;
    bool backPropagating;



private://generated attribute

    std::map <layerCoordinate, std::vector <size_t>, layerCoordinateCmp> layersCumulatedDimensions;
    std::map <layerCoordinate, size_t, layerCoordinateCmp> layersTotalNumberOfNeuron;

};








template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::createLink(neuronCoordinate &c1, neuronCoordinate &c2, double initialWeight)
{
    assert(neuronCoordinateCmp::compLayer(c1, c2));
    assert(!links[c1].count(c2));
    links[c1][c2] = initialWeight;
    neurons[c1.first][c1.second].linked = 1;
    neurons[c2.first][c2.second].linked = 1;
    auto toPush = std::make_pair(&neurons[c2.first][c2.second]
        , links[c1][c2]);
    neurons[c1.first][c1.second].next.push_back(toPush);
    toPush.first = &neurons[c1.first][c1.second];
    neurons[c2.first][c2.second].previous.push_back(toPush);
}


/*
//no delete link. Only object destructor.
template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::deleteLink(neuronCoordinate &c1, neuronCoordinate &c2)
{
    assert(links[c1].count(c2));
    links[c1].erase(c2);
}


*/



void neuralNetwork::toName()
{
    neuron()
}

















template <typename ExtraDataT>
neuralNetwork<ExtraDataT>::neuralNetwork()
{
}










#endif // NEURALNETWORK_H
