#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


///This library only support DAG type neural network. For instance, it doesnt support recurrent neural network.

///too frequents agregations decrease efficiency and precision of recurrent network, especially the one with link to old self-history



///be careful. ++(incrementing operator) for first feeding layer receiving input
/// calls forwartdCompute which add a bias and apply activation function






///eventuellement ajouter ultérieurement un quatrième paramètre à l'altération inter cycle de neurone, qui serait le numéro de la copie du neural network
/// qui permettrait par exemple de mettre les entraînements subsidaires qui changent peu les coefficients, dans un même groupe de thread et dans un même neural network
/// et de faire savoir qu'il s'agit de ce type de thread aux callback (calculateurs et feeders)



///pour la version classique et la version à plusieurs feed par network à 2 couches d'intervalles, travailler uniquement dans les copies (et faire des agrégations régulières)
/// comme ça quand on souhaite arrêter l'entraînement, il suffit de détruire les copies (éventuellement faire une dernière agrégation), et tout est normalisé



///normalize callback is the inter half computation normalization. Think of changing name, so we can use this name.

///evaluate training end and neuron reset callback





///srand must be called  if the randomness has tobe controlled (here rand() is called)



///dont use the main neural network for the training






#include <map>
#include <vector>
#include <mutex>
#include <cassert>
#include <cmath>
#include <initializer_list>



#include "neuron.h"



namespace jo_nn
{



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
    bool droped;
    size_t historySize;
    ExtraDataT ExtraData_p;
};








typedef std::pair <layerCoordinate, size_t> neuronCoordinate;


/*template <typename ExtraDataT>
using neuronConfigureFunction = std::function <neuronConstructorParameters<ExtraDataT>(const neuronCoordinate &n)>;*/



typedef std::pair<std::pair<size_t, size_t>, double/*initial weight*/> layerConnections;

typedef std::function <std::vector<layerConnections>(std::vector <size_t>, std::vector<size_t>)> layersConnectFunction;





class layerCoordinateCmp
{
public:
    inline bool operator()(layerCoordinate const&a, layerCoordinate const&b) const
    {
        return a < b;
    }
};

class neuronCoordinateCmp
{
public:
    inline bool operator()(neuronCoordinate const&a, neuronCoordinate const&b) const
    {
        if(a.first == b.first)
            return a.second < b.second;
        else
        {
            return a.first < b.first;
        }
    }
    static inline bool compLayer(neuronCoordinate const&a, neuronCoordinate const&b)
    {
        if(a.first < b.first)
            return 1;
        else
            return 0;
    }
};




typedef std::map<layerCoordinate, std::vector <double>, layerCoordinateCmp> layerFeed;




template <typename ExtraDataT>
using interComputationNeuronAlterationFunction = std::function<void(neuron<ExtraDataT>* target, neuronCoordinate c, size_t cycle)>;//necessitate to use the unique id giver wrapper (not created yet)
//so cycle is unique






///check that c_interComputationNeuronAlterationFunction is not null before calling





template <//typename T/*ptr on structure that will be passed on assertion*/,
typename ExtraDataT/*on neurons*/>
class neuralNetwork
{

public:

    neuralNetwork();
    ~neuralNetwork() = default;
    neuralNetwork(neuralNetwork &);//la copie prends un non const car on try lock le mutex (car c'est la seule manière de voire son état)

    neuralNetwork(neuralNetwork&&) = delete;
    neuralNetwork& operator=(const neuralNetwork&) = delete;
    neuralNetwork& operator=(neuralNetwork&&) = delete;



    void createLink(const neuronCoordinate &c1, const neuronCoordinate &c2, double initialWeight);


    ///void createRecursiveLink(neuronCoordinate &c1, size_t recursion_level/*0 = previous*/, double initialWeight);
    ///to add


    //void build();
    void refresh();


    ///JUST BELOW :
    void addLayer(std::vector <size_t>, layerCoordinate lc = 0, bool input = 0, bool output = 0, neuronConfigureFunction<ExtraDataT> = /*defaultRelu<ExtraDataT>*/0);
    void addLayer(std::initializer_list <size_t>, layerCoordinate lc = 0, bool input = 0, bool output = 0, neuronConfigureFunction<ExtraDataT> = /*defaultRelu<ExtraDataT>*/0);

    void alterLayer(layerCoordinate lc, neuronConfigureFunction<ExtraDataT>, bool input = 0, bool output = 0);


    void connectLayers(layerCoordinate feedingLayer, layerCoordinate fedLayer
        ,  layersConnectFunction connections);


    ///void connectLayersRecursive(layerCoordinate feedingLayer
    ///    , layersConnectFunction connections);*/
    ///to add



    void loadCoeffs(std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp>);//doesn't handle recurrent coeffs
    std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp> dumpCoeff();//doesn't handle recurrent coeffs









public://Fonctions de calcul

    static size_t toIndex(std::vector <size_t> index, std::vector <size_t> dims);
    static std::vector <size_t> toCoordinate(size_t index, std::vector <size_t> dims);

    static size_t totalSize(std::vector <size_t> dims);
    static std::vector <size_t> reverseOrderCumulativeSize(std::vector <size_t> dims);


private://(Sous)Fonctions de construction
    void build();




private://Variables organisatrices à partir desquelles on génère le reste.


    std::map <layerCoordinate, std::vector <neuron<ExtraDataT>>, layerCoordinateCmp>
        neurons;

public :std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp>
        links;///make this section private again when debugging is finished
    std::map <neuronCoordinate, std::map <size_t, double>, neuronCoordinateCmp>
        selfLinks;

    std::map <layerCoordinate, std::vector <size_t>, layerCoordinateCmp>
        dimensions;
    std::map <layerCoordinate, std::pair<bool, bool>, layerCoordinateCmp>
        layersInformationInputOutput;






private://generated attributes

    std::map <layerCoordinate, std::vector <size_t>, layerCoordinateCmp> layersCumulatedDimensions;
    std::map <layerCoordinate, size_t, layerCoordinateCmp> layersTotalNumberOfNeuron;










public:

    interComputationNeuronAlterationFunction<ExtraDataT> c_interComputationNeuronAlterationFunction;///change constructor to initialize to an empty inline function
    void setInterComputationNeuronAlterationFunction(interComputationNeuronAlterationFunction<ExtraDataT>);


private://computation time Attribute

    double errorIndicator;//éventuellement ajouter un historique
    bool backPropagating;
    size_t cycle;//not that this cycle only take account of the current neural network, in case of multithread
    std::mutex computing;


public://computing

    layerFeed assertion(layerFeed input);//default is forwardState

    std::vector <neuralNetwork<ExtraDataT>*> getBindNeuralNetwork(size_t n);

    layerFeed forCompute(layerFeed input);
    void backCompute(layerFeed input, double errorIndicator);

//private:
    layerFeed internal_assertion(layerFeed input);
};







template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::createLink(const neuronCoordinate &c1, const neuronCoordinate &c2, double initialWeight)//to modify to add recursive connection
{
    //assert(!layerCoordinateCmp(c2.first, c1.first));
    assert(neuronCoordinateCmp::compLayer(c1, c2));//non recursive

    auto &a = links[c1];
    assert(!a.count(c2));

    links[c1][c2] = initialWeight;
    neurons[c1.first][c1.second].linked = 1;
    neurons[c2.first][c2.second].linked = 1;
    std::pair <neuron<ExtraDataT>*, double*> toPush;
    toPush = std::make_pair(&neurons[c2.first][c2.second]
        , &links[c1][c2]);
    neurons[c1.first][c1.second].next.push_back(toPush);
    toPush.first = &neurons[c1.first][c1.second];
    neurons[c2.first][c2.second].previous.push_back(toPush);
}








//pour construire le vecteur de neurones consécutifs pour le calcul avec les repères de synchronisation, faire appel à un callback
//éventuellement en utiliser un autre pour le découpage (ou réserver ça à la version plus modulaire ultérieure)



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

template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::addLayer(std::vector <size_t> dims, layerCoordinate lc, bool input, bool output, neuronConfigureFunction<ExtraDataT> f)
{
    if(!f)
        f = defaultRelu<ExtraDataT>;
    size_t upBound = totalSize(dims);
    if((!lc))
    {
        if(!neurons.empty())
        {
            lc = (--neurons.end())->first;
            lc++;
        }
        else
            lc = 0;
    }
    dimensions[lc] = dims;
    layersInformationInputOutput[lc].first = input;
    layersInformationInputOutput[lc].second = output;
    layersTotalNumberOfNeuron[lc] = totalSize(dims);
    layersCumulatedDimensions[lc] = reverseOrderCumulativeSize(dims);
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
            , param.droped
            , param.historySize

            , &cycle
            , &errorIndicator
            , &backPropagating
            , std::make_pair(lc, i)

            , param.ExtraData_p);
    }
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::addLayer(std::initializer_list <size_t> d, layerCoordinate lc, bool input, bool output, neuronConfigureFunction<ExtraDataT> f)
{
    return addLayer(std::vector <size_t>(d.begin(), d.end()), lc, input, output, f);
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::alterLayer(layerCoordinate lc, neuronConfigureFunction<ExtraDataT> f, bool input, bool output)
{
    layersInformationInputOutput[lc].first = input;
    layersInformationInputOutput[lc].second = output;
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

              , &cycle
              , &errorIndicator
              , &backPropagating
              , std::make_pair(lc, i)

              , param.ExtraData_p);
    }
}






template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::connectLayers(layerCoordinate feedingLayer, layerCoordinate fedLayer///non recursive
    , layersConnectFunction connections)
{
    auto c = connections(dimensions[feedingLayer], dimensions[fedLayer]);
    for(auto a : c)
        createLink(std::make_pair(feedingLayer, a.first.first)
            , std::make_pair(fedLayer, a.first.second)
            , a.second);
}






/*
template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::connectLayersRecursive(layerCoordinate feedingLayer
    , layersConnectFunction connections)
{
    for(auto a : connections)
        createLink(std::make_pair(a.first.first, feedingLayer));
}
*/




template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::build()///implement
{

}






template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::setInterComputationNeuronAlterationFunction(interComputationNeuronAlterationFunction<ExtraDataT> f)
{
    c_interComputationNeuronAlterationFunction = f;
}






template <typename ExtraDataT>
size_t neuralNetwork<ExtraDataT>::toIndex(std::vector <size_t> index, std::vector <size_t> dims)
{
    size_t ret = 0;
    for(size_t i = 0; i < index.size(); ++i)
        ret += index[i] * dims[i];
    return ret;
}


template <typename ExtraDataT>
std::vector <size_t> neuralNetwork<ExtraDataT>::toCoordinate(size_t index, std::vector <size_t> dims)
{
    std::vector <size_t> ret(dims.size());
    for(auto a : dims)
    {
        ret.push_back(index / a);
        index %= a;
    }
    return ret;
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
std::vector <size_t> neuralNetwork<ExtraDataT>::reverseOrderCumulativeSize(std::vector <size_t> dims)
{
    std::vector <size_t> ret(dims.size(), 1);
    size_t j = 0;
    for(size_t i = dims.size() - 2; i + 1; --i, j++)
        ret[i] = ret[i + 1] * dims[j];
    return ret;

}




template <typename ExtraDataT>
neuralNetwork<ExtraDataT>::neuralNetwork(neuralNetwork& nn):
neurons(nn.neurons),
dimensions(nn.dimensions),
layersInformationInputOutput(nn.layersInformationInputOutput),
layersCumulatedDimensions(nn.layersCumulatedDimensions),
layersTotalNumberOfNeuron(nn.layersTotalNumberOfNeuron),
c_interComputationNeuronAlterationFunction(nn.c_interComputationNeuronAlterationFunction)
{
    assert(nn.computing.try_lock());
    nn.computing.unlock();
    cycle = 0;
    errorIndicator = 0;
    backPropagating = 0;
    for(auto a : nn.links)
        for(auto b : a.second)
                createLink(a.first, b.first, b.second);
}







template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::loadCoeffs(std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp> i)
{
    links.clear();
    links = i;
}

template <typename ExtraDataT>
std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp> neuralNetwork<ExtraDataT>::dumpCoeff()
{
    return links;
}



/*
void callback_sucessiveLayers(std::function<std::pair<layerFeed, T*>(size_t, size_t)> feedFor
, std::function<layerFeed(size_t, size_t, std::pair<layerFeed, T*>)> feedBac
*/





template <typename ExtraDataT>
neuralNetwork<ExtraDataT>::neuralNetwork():
c_interComputationNeuronAlterationFunction(/*emptyInterComputationNeuronAlterationFunction<ExtraDataT>*/0)
{
    cycle = 0;
    errorIndicator = 0;
    backPropagating = 0;
}



//faire deux versions : une versions où on fournit des callbacks d'assertion/apprentissage et une où on demande deux callback : de feedFor et feedBac




template <typename ExtraDataT>
layerFeed neuralNetwork<ExtraDataT>::assertion(layerFeed input)
{
    neuralNetwork t(*this);
    return t.internal_assertion(input);
}












template <typename ExtraDataT>
layerFeed neuralNetwork<ExtraDataT>::internal_assertion(layerFeed input)
{
    layerFeed ret;
    assert(computing.try_lock());
    build();

    for(auto &a : neurons)
        for(auto &b : a.second)
            b();//pourquoi ça ne resettais pas les valeurs ??

    for(std::pair<layerCoordinate, std::vector <double>> a : input)
    {
        assert(layersInformationInputOutput[a.first].first);
        for(size_t i = 0; i < a.second.size(); ++i)
            neurons[a.first][i].forwardValue = a.second[i];
    }

    for(auto/*std::pair <layerCoordinate, std::vector <neuron<ExtraDataT>>>*/ &a : neurons)
        /*for(size_t i = 0; i < a.second.size(); i++)*/for(neuron<ExtraDataT> &b : a.second)
            /*a.second[i]++;*/b++;

    for(std::pair <layerCoordinate, std::vector <neuron<ExtraDataT>>> a : neurons)
    {
        if(!(layersInformationInputOutput[a.first].second))
            continue;
        for(neuron<ExtraDataT> &b : a.second)
            ret[a.first].push_back(b.forwardValue);
    }

    build();
    computing.unlock();
    return ret;
}











template <typename ExtraDataT>
std::vector <neuralNetwork<ExtraDataT>*> neuralNetwork<ExtraDataT>::getBindNeuralNetwork(size_t n)
{
    std::vector <neuralNetwork<ExtraDataT>*> ret;
    for(size_t i = 0; i < n; ++i)
        ret.push_back(new neuralNetwork<ExtraDataT>(*this));
    for(auto &d : ret)
    {
        for(auto &vec : d->neurons)
        {
            for(auto &neur : vec.second)
            {
                for(auto &link : neur.next)
                {
                    link.second = &(links[neur.nCoordinate][link.first->nCoordinate]);
                }
                for(auto &link : neur.previous)
                {
                    link.second = &(links[link.first->nCoordinate][neur.nCoordinate]);
                }
            }
        }
    }
    return ret;
}


template <typename ExtraDataT>
layerFeed neuralNetwork<ExtraDataT>::forCompute(layerFeed input)
{
    layerFeed ret;

    build();


    assert(computing.try_lock());
    backPropagating = 0;




    for(std::pair<layerCoordinate, std::vector <double>> a : input)
    {
        assert(layersInformationInputOutput[a.first].first);
        for(size_t i = 0; i < a.second.size(); ++i)
            neurons[a.first][i].forwardValue = a.second[i];
    }

    for(auto/*std::pair <layerCoordinate, std::vector <neuron<ExtraDataT>>>*/ &a : neurons)
        /*for(size_t i = 0; i < a.second.size(); i++)*/for(neuron<ExtraDataT> &b : a.second)
            /*a.second[i]++;*/b++;

    for(std::pair <layerCoordinate, std::vector <neuron<ExtraDataT>>> a : neurons)
    {
        if(!(layersInformationInputOutput[a.first].second))
            continue;
        for(neuron<ExtraDataT> &b : a.second)
            ret[a.first].push_back(b.forwardValue);
    }


    for(auto &a : neurons)
        for(auto &b : a.second)
            b();


    computing.unlock();


    build();

    return ret;
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::backCompute(layerFeed backInput, double errorIndicator_p)
{
    build();


    assert(computing.try_lock());
    backPropagating = 1;
    errorIndicator = errorIndicator_p;


    for(auto &a : backInput)
    {
        assert(layersInformationInputOutput[a.first].second);
        for(size_t i = 0; i < a.second.size(); ++i)
            neurons[a.first][i].backwardValue = a.second[i];
    }


    for(auto a = neurons.end(); a != neurons.begin(); )
    {
        a--;
        for(auto &b : a->second)
            b--;
    }

    for(auto &a : neurons)
        for(auto &b : a.second)
            b();


    computing.unlock();
    ++cycle;


    build();
}




}


#endif // NEURALNETWORK_H
