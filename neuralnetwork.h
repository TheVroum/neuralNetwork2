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
    std::function <void(bool direction, neuron<ExtraDataT>* target)> c_normalize_p;
    std::function <double(double input)> c_activationFunction_p;
    std::function <double(double input)> c_activationFunctionDerivative_p;
    neuronCoeffDerivativeCalculatorFunction c_coeffDerivativeCalculator_p;
    computationFunction<ExtraDataT> forwardCalculator;
    computationFunction<ExtraDataT> backwardCalculator;
    double bias_p;
    size_t historySize;
    size_t selfCoeffsNumber;
    ExtraDataT ExtraData_p;
};



typedef std::pair <size_t, size_t> layerCoordinate;
typedef std::pair <layerCoordinate, size_t> neuronCoordinate;


template <typename ExtraDataT>
using neuronConfigureFunction = std::function <neuronConstructorParameters<ExtraDataT>(const neuronCoordinate &n)>;


template <typename ExtraDataT>
class neuralNetwork
{

public:

    neuralNetwork() = default;
    ~neuralNetwork() = default;

    neuralNetwork(const neuralNetwork&) = delete;
    neuralNetwork(neuralNetwork&&) = delete;
    neuralNetwork& operator=(const neuralNetwork&) = delete;
    neuralNetwork& operator=(neuralNetwork&&) = delete;



    void createLink(neuronCoordinate &c1, neuronCoordinate &c2, double initialWeight);


    void createRecursiveLink(neuronCoordinate &c1, size_t recursion_level/*0 = previous*/, double initialWeight);


    //void build();
    void refresh();


    ///JUST BELOW :
    void addLayer(std::vector <size_t>, layerCoordinate lc/* = {0, 0}*/ /*uncomment and add third default parameter*/, neuronConfigureFunction<ExtraDataT>);
    void addLayer(std::initializer_list <size_t>, layerCoordinate lc/* = {0, 0}*/, neuronConfigureFunction<ExtraDataT>);

    void alterLayer(layerCoordinate lc, neuronConfigureFunction<ExtraDataT>);


    void connectLayers(layerCoordinate feedingLayer, layerCoordinate fedLayer
        , std::function <std::vector<std::pair<size_t, size_t>>()> connectionCallback);





public://Fonctions de calcul

    static size_t toIndex(std::vector <size_t> index, std::vector <size_t> dims);
    static std::vector <size_t> toCoordinate(size_t index, std::vector <size_t> dims);
    static size_t totalSize(std::vector <size_t> dims);


private://(Sous)Fonctions de construction
void buildDimensions();




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


private://computation time Attribute

    double errorIndicator;
    bool backPropagating;
    size_t cycle;



private://generated attributes

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








//pour construire le vecteur de neurones consécutifs pour le calcul avec les repères de synchronisation, faire appel à un callback
//éventuellement en utiliser un autre pour le découpage (ou réserver ça à la version plus modulaire ultérieure)




template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::addLayer(std::vector <size_t> dims, layerCoordinate lc, neuronConfigureFunction<ExtraDataT> f)
{
    size_t upBound = totalSize(dims);
    if((!lc.first) && (!lc.second))
        lc = (neurons.end--)->first;
    lc.first++;
    std::vector <neuron<ExtraDataT>> &v = neurons[lc];
    for(size_t i = 0; i < upBound; ++i)
    {
        auto param = f(std::make_pair(lc, i));
        v.emplace_back(param.c_normalize_p
            , param.c_activationFunction_p
            , param.c_activationFunctionDerivative_p
            , param.c_coeffDerivativeCalculator_p
            , param.forwardCalculator
            , param.backwardCalculator
            , param.bias_p
            , param.historySize
            , param.selfCoeffsNumber

            , &cycle
            , &errorIndicator
            , &backPropagating
            , std::make_pair(lc, i)

            , param.ExtraData_p);
    }
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::addLayer(std::initializer_list <size_t> d, layerCoordinate lc, neuronConfigureFunction<ExtraDataT> f)
{
    return addLayer(std::vector <size_t>(d.begin(), d.end()), lc, f);
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::alterLayer(layerCoordinate lc, neuronConfigureFunction<ExtraDataT> f)
{
    for(size_t i = 0; i < neurons[lc].size(); ++i)
    {
        auto param = f(std::make_pair(lc, i));
        neurons[lc].set(param.c_normalize_p
              , param.c_activationFunction_p
              , param.c_activationFunctionDerivative_p
              , param.c_coeffDerivativeCalculator_p
              , param.forwardCalculator
              , param.backwardCalculator
              , param.bias_p
              , param.historySize
              , param.selfCoeffsNumber

              , &cycle
              , &errorIndicator
              , &backPropagating
              , std::make_pair(lc, i)

              , param.ExtraData_p);
    }
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::connectLayers(layerCoordinate feedingLayer, layerCoordinate fedLayer
    , std::function <std::vector<std::pair<size_t, size_t>>()> connectionCallback)
{

}









template <typename ExtraDataT>
size_t neuralNetwork<ExtraDataT>::toIndex(std::vector <size_t> index, std::vector <size_t> dims)
{

}

template <typename ExtraDataT>
std::vector <size_t> neuralNetwork<ExtraDataT>::toCoordinate(size_t index, std::vector <size_t> dims)
{

}

template <typename ExtraDataT>
size_t neuralNetwork<ExtraDataT>::totalSize(std::vector <size_t> dims)
{
    size_t ret = 1;
    for(auto a : dims)
        ret *= a;
    return ret;
}




template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::buildDimensions()
{
    layersCumulatedDimensions[dimensions.end()--->first] = 1;
    for(auto &a : dimensions)
    {/*
        for(int i = dim.size() - 2; i + 1; i--)
            tailles[i] = tailles[i + 1] * dim[i + 1];*/
    }
}








#endif // NEURALNETWORK_H
