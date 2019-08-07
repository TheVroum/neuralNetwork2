#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


///Attention à ne pas "forwardcomputer" les layers qui feedent

///je suis en train d'ajouter un vecteur de neuralnetwork* en cas de double appel à getbinneuralnetwork, qui supprimerai les objets
///et ajouter un size_t qui indique quelle copie on est en train d'utiliser. Fait. Faut il vraiment supprimer automatiquement les objets créés lors d'un appel précédent



///eventuellement ajouter ultérieurement un quatrième paramètre à l'altération inter cycle de neurone, qui serait le numéro de la copie du neural network
/// qui permettrait par exemple de mettre les entraînements subsidaires qui changent peu les coefficients, dans un même groupe de thread et dans un même neural network
/// et de faire savoir qu'il s'agit de ce type de thread aux callback (calculateurs et feeders)




///normalize callback is the inter half computation normalization. Think of changing name, so we can use this name.





///srand must be called  if the randomness has to be controlled (here rand() is called)



///dont use the main neural network for the training///Pourquoi ?






#include <map>
#include <vector>
#include <mutex>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <new>







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
    ExtraDataT ExtraData_p;
};


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
using interComputationNeuronAlterationFunction = std::function<void(neuron<ExtraDataT>* target, neuronCoordinate c, double errorIndicator, size_t cycle)>;//necessitate to use the unique id giver wrapper (not created yet)
//so cycle is unique

///check that c_interComputationNeuronAlterationFunction is not null before calling


template <typename ExtraDataT>
using neuronConfigureFunction = std::function <neuronConstructorParameters<ExtraDataT>(const neuronCoordinate n, const std::vector<size_t>*dimensionsOfTheLayer)>;


template <typename ExtraDataT/*on neurons*/>///One neural network should not be acessed simultaneously from different thread. For multi threading use get_bind_neuralnetwork().
class neuralNetwork///mettre ensuite tous les attributs en privés
{

public://Constructeurs et destructeur

    neuralNetwork();
    ~neuralNetwork();

public:
    neuralNetwork(neuralNetwork &);//la copie prends un non const car on try lock le mutex (car c'est la seule manière de voire son état)

    neuralNetwork(neuralNetwork&&) = delete;
    neuralNetwork& operator=(const neuralNetwork&) = delete;
    neuralNetwork& operator=(neuralNetwork&&) = delete;

public://Fonctions de construction

    void addLayer(std::vector <size_t>, layerCoordinate lc = 0, bool input = 0, bool output = 0, neuronConfigureFunction<ExtraDataT> = /*defaultRelu<ExtraDataT>*/0);
    void addLayer(std::initializer_list <size_t>, layerCoordinate lc = 0, bool input = 0, bool output = 0, neuronConfigureFunction<ExtraDataT> = /*defaultRelu<ExtraDataT>*/0);

    void createLink(const neuronCoordinate &c1, const neuronCoordinate &c2, double initialWeight);

    void connectLayers(layerCoordinate feedingLayer, layerCoordinate fedLayer
        ,  layersConnectFunction connections);

public://Fonctions de construction et d'altération

    void setInterComputationNeuronAlterationFunction(interComputationNeuronAlterationFunction<ExtraDataT>);

public://Fonctions d'altération

    void alterLayer(layerCoordinate lc, neuronConfigureFunction<ExtraDataT>, bool input = 0, bool output = 0);

    void deleteLink(const neuronCoordinate &c1, const neuronCoordinate &c2);
    void deleteAllLinks();
    //pas de delete links between layers pour l'instant

public://Fonctions d'actualisation devant être appelé dans beaucoup de situations
    //void refresh();


public://Fonctions d'altération

    void loadCoeffs(std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp>);//doesn't handle recurrent coeffs
    std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp> dumpCoeff();//doesn't handle recurrent coeffs

public://Fonctions de calcul

    static size_t toIndex(std::vector <size_t> index, std::vector <size_t> dims);
    static std::vector <size_t> toCoordinate(size_t index, std::vector <size_t> dims);

    static size_t totalSize(std::vector <size_t> dims);
    static std::vector <size_t> reverseOrderCumulativeSize(std::vector <size_t> dims);


private://(Sous)Fonctions de construction///ici
    void build();///in refresh

public://computing

    layerFeed assertion(layerFeed input);//default is forwardState

    std::vector <neuralNetwork<ExtraDataT>*> getBindNeuralNetwork(size_t n);
    //void destroyBindNeuralNetwork(std::vector <neuralNetwork<ExtraDataT>*>);//automatically done

    layerFeed forCompute(layerFeed input);
    void backCompute(layerFeed input, double errorIndicator);



private://Variables organisatrices à partir desquelles on génère le reste.


    std::map <layerCoordinate, std::vector <neuron<ExtraDataT>>, layerCoordinateCmp>
        neurons;
public://to delete
    std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp>
        links;
private://to delete
    std::map <layerCoordinate, std::vector <size_t>, layerCoordinateCmp>
        dimensions;
    std::map <layerCoordinate, std::pair<bool, bool>, layerCoordinateCmp>
        layersInformationInputOutput;

    interComputationNeuronAlterationFunction<ExtraDataT> c_interComputationNeuronAlterationFunction;///change constructor to initialize to an empty inline function

    neuralNetwork *parent;
    
    int copyNumber;//Or number of childs if parent = nullptr


public://private://Variables générées

    std::map <neuronCoordinate,
        std::vector <neuronCoordinate>> nextCoord, previousCoord;
    std::map <neuronCoordinate,
        std::vector <std::pair <neuron<ExtraDataT>*, double*>>> next, previous;
    std::vector <std::pair<neuron<ExtraDataT>*, neuronCoordinate>> vectNeuronsFor, vectNeuronsBac, vectNeuronsAll;//vectoriser les neurones


    std::map <layerCoordinate, std::vector <size_t>, layerCoordinateCmp> layersCumulatedDimensions;
    std::map <layerCoordinate, size_t, layerCoordinateCmp> layersTotalNumberOfNeuron;

    std::vector <neuralNetwork<ExtraDataT>*> boundNeuralNetwork;

private://Variable générée mais à initialiser dans le constructeur
    bool built;


private://computation time Attribute
    double errorIndicator;//éventuellement ajouter un historique
    bool backPropagating;
    size_t cycle;//not that this cycle only take account of the current neural network, in case of multithread
    //also note that cycle might be different accross bound neural networks

    ///std::mutex computing;

};



//this is an exception among construction/alteration function as it calls another so flags (like built) and other things that shiould be done are not here, and delegated
template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::addLayer(std::initializer_list <size_t> d, layerCoordinate lc, bool input, bool output, neuronConfigureFunction<ExtraDataT> f)
{
    return addLayer(std::vector <size_t>(d.begin(), d.end()), lc, input, output, f);
}





template <typename ExtraDataT, int bias = 0/*millionieme*/, size_t learningRate/*millionieme*/ = 10000>
neuronConstructorParameters<ExtraDataT> defaultRelu(const neuronCoordinate, const std::vector<size_t>*);

template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::addLayer(std::vector <size_t> dims, layerCoordinate lc, bool input, bool output, neuronConfigureFunction<ExtraDataT> f)
{
    built = 0;
    if(!f)//Faire en sorte que les neurones soient bien initialisés d'une manière ou d'une autre, même si l'utilisateur ne choisit pas de fonction d'initialisation particulière
        f = (defaultRelu<ExtraDataT>);

    if((!lc))//La coordonnées 0 signifie mettre la layer à la suite. La coordonnées choisie est la coordonnées de la dernière couche, incrémentée d'une unité divisible.
    {
        if(!neurons.empty())
        {
            lc = (--neurons.end())->first;
            lc++;
        }
        else
            lc = layerCoordinate(0);//Si le neuralNetwork (abrégé nn) est vide, alors on laisse la coordonnée 0
    }

    dimensions[lc] = dims;
    layersInformationInputOutput[lc].first = input;
    layersInformationInputOutput[lc].second = output;

    std::vector <neuron<ExtraDataT>> &v = neurons[lc];//La map construit l'objet car il n'existe pas
    size_t upBound = totalSize(dims);//Le produit des dimensions nous donne le nombre total de neurone (hyper paralépipède droit). Le tout est contenu dans un vecteur d'une dimension
    for(size_t i = 0; i < upBound; ++i)
    {
        auto param = f(std::make_pair(lc, i), &dimensions[lc]);
        v.emplace_back(param.c_normalize_p
            , param.c_activationFunction_p
            , param.c_activationFunctionDerivative_p
            , param.c_coeffDerivativeCalculator_p
            , param.forwardCalculator
            , param.backwardCalculator
            , param.bias_p
            , param.droped

            , param.ExtraData_p);
    }
}




template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::createLink(const neuronCoordinate &c1, const neuronCoordinate &c2, double initialWeight)//to modify to add recursive connection
{
    built = 0;
    assert(neuronCoordinateCmp::compLayer(c1, c2));

    auto &a = links[c1];
    assert(!a.count(c2));//Ne permet de double connection entre deux neurones donnés
    links[c1][c2] = initialWeight;
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::connectLayers(layerCoordinate feedingLayer, layerCoordinate fedLayer///non recursive
    , layersConnectFunction connections)
{
    built = 0;
    auto c = connections(dimensions[feedingLayer], dimensions[fedLayer]);
    for(auto a : c)
        createLink(std::make_pair(feedingLayer, a.first.first)
            , std::make_pair(fedLayer, a.first.second)
            , a.second);
}



template <typename ExtraDataT>
neuralNetwork<ExtraDataT>::~neuralNetwork()
{
    if(boundNeuralNetwork.size())
        for(auto a : boundNeuralNetwork)
            delete a;
}


template <typename ExtraDataT>//this function doesn't set built to 0, as it doesnt require rebuild
void neuralNetwork<ExtraDataT>::setInterComputationNeuronAlterationFunction(interComputationNeuronAlterationFunction<ExtraDataT> f)
{
    c_interComputationNeuronAlterationFunction = f;
}






template <typename ExtraDataT>//cette fonction ne prévoir pas de changer les dimensions (je n'en vois pas l'utilité...)
void neuralNetwork<ExtraDataT>::alterLayer(layerCoordinate lc, neuronConfigureFunction<ExtraDataT> f, bool input, bool output)
{
    built = 0;
    layersInformationInputOutput[lc].first = input;
    layersInformationInputOutput[lc].second = output;
    for(size_t i = 0; i < neurons[lc].size(); ++i)
    {
        auto param = f(std::make_pair(lc, i), dimensions[lc]);
        neurons[lc][i] = neuron<ExtraDataT>(param.c_normalize_p
              , param.c_activationFunction_p
              , param.c_activationFunctionDerivative_p
              , param.c_coeffDerivativeCalculator_p
              , param.forwardCalculator
              , param.backwardCalculator
              , param.bias_p
              , param.historySize

              , param.ExtraData_p);
    }
}





template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::deleteLink(const neuronCoordinate &c1, const neuronCoordinate &c2)
{
    built = 0;
    auto &a = links[c1];//crée la map si elle n'existe pas
    assert(!a.count(c2));//Vérifie que la connexion existe bien avant de la supprimer
    a.erase(c2);
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::deleteAllLinks()
{
    built = 0;
    links.clear();
}





template <typename ExtraDataT>//doesnt require rebuilding if the topology is conserved. This function doesnt force it so rebuilding is systematically retriggered (en différé)
void neuralNetwork<ExtraDataT>::loadCoeffs(std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp> i)
{
    built = 0;
    for(auto a : i)
        for(auto b : a.second)
        {
            //ces assertions sont strictement requises. Cependant elle ne permettent pas de changement de "topologies" des connexions
            //On peut sinon itérer en parallèle et vérifier que les coordonnées snt les mêmes

            assert(a.first.first < b.first.first);

            assert(neurons.count(a.first.first));
            assert(neurons[a.first.first].size() > a.first.second);

            assert(neurons.count(b.first.first));
            assert(neurons[b.first.first].size() > b.first.second);
        }
    links.clear();
    links = i;
}

template <typename ExtraDataT>
std::map <neuronCoordinate, std::map <neuronCoordinate, double, neuronCoordinateCmp>, neuronCoordinateCmp> neuralNetwork<ExtraDataT>::dumpCoeff()
{
    return links;
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
links(),//links wont be used as the used links are those of the parent
dimensions(nn.dimensions),
layersInformationInputOutput(nn.layersInformationInputOutput),
c_interComputationNeuronAlterationFunction(nn.c_interComputationNeuronAlterationFunction),
parent(&nn),
copyNumber(nn.copyNumber),
built(0),
errorIndicator(0),
backPropagating(0),
cycle(0)
{
    assert(nn.parent == nullptr);
    ///nn.computing.try_lock();
    nn.copyNumber = nn.copyNumber + 1;
    ///nn.computing.unlock();
    build();
}


template <typename ExtraDataT>
inline void emptyInterComputationNeuronAlterationFunction
    (neuron<ExtraDataT>*, neuronCoordinate, double, size_t);


template <typename ExtraDataT>
neuralNetwork<ExtraDataT>::neuralNetwork():
c_interComputationNeuronAlterationFunction(emptyInterComputationNeuronAlterationFunction<ExtraDataT>),
parent(nullptr),
copyNumber(0),
built(0),
errorIndicator(0),
backPropagating(0),
cycle(0)
{
    ///computing.try_lock();
    ///computing.unlock();
}



//faire deux versions : une versions où on fournit des callbacks d'assertion/apprentissage et une où on demande deux callback : de feedFor et feedBac




template <typename ExtraDataT>
layerFeed neuralNetwork<ExtraDataT>::assertion(layerFeed input)
{
    if(!built)
        build();
    neuralNetwork t(*this);
    return t.forCompute(input);
}











std::vector <void*> vv;

template <typename ExtraDataT>
std::vector <neuralNetwork<ExtraDataT>*> neuralNetwork<ExtraDataT>::getBindNeuralNetwork(size_t n)
{
    assert(!parent);

    //block optionnel
    if(boundNeuralNetwork.size())
        for(auto a : boundNeuralNetwork)
            delete a;
    //fin de block optionnel

    std::vector <neuralNetwork<ExtraDataT>*> ret;
    for(size_t i = 0; i < n; ++i)
        ret.push_back(new neuralNetwork<ExtraDataT>(*this))
                , vv.push_back(reinterpret_cast<void*>(&(ret.back()->backPropagating)));

    for(auto a : ret)
        boundNeuralNetwork.push_back(a);

    return ret;
}


template <typename ExtraDataT>
layerFeed neuralNetwork<ExtraDataT>::forCompute(layerFeed input)
{
    if(!built)
        build();
    layerFeed ret;

    ///assert(computing.try_lock());
    backPropagating = 0;




    for(std::pair<layerCoordinate, std::vector <double>> a : input)
    {
        assert(layersInformationInputOutput[a.first].first);
        for(size_t i = 0; i < a.second.size(); ++i)
            neurons[a.first][i].forwardValue = a.second[i];
    }

    for(auto &a : vectNeuronsFor)
        a.first->forC(next[a.second], previous[a.second], a.second, errorIndicator, cycle);

    for(auto &a : neurons)
    {
        if(!(layersInformationInputOutput[a.first].second))
            continue;
        for(neuron<ExtraDataT> &b : a.second)
            ret[a.first].push_back(b.forwardValue);
    }


    for(auto &b : vectNeuronsAll)
            (*(b.first))(backPropagating);


    ///computing.unlock();

    return ret;
}


template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::backCompute(layerFeed backInput, double errorIndicator_p)
{
    if(!built)
        build();
    ///assert(computing.try_lock());
    backPropagating = 1;
    errorIndicator = errorIndicator_p;


    for(auto &a : backInput)
    {
        assert(layersInformationInputOutput[a.first].second);
        for(size_t i = 0; i < a.second.size(); ++i)
            neurons[a.first][i].backwardValue = a.second[i];
    }


    for(size_t i = vectNeuronsBac.size() - 1; i + 1; --i)
    {
        vectNeuronsBac[i].first->backC(next[vectNeuronsBac[i].second]
            , previous[vectNeuronsBac[i].second], vectNeuronsBac[i].second
            , errorIndicator, cycle);
    }




    for(auto &b : vectNeuronsAll)
            (*(b.first))(backPropagating);


    ///computing.unlock();
    ++cycle;
}





template <typename ExtraDataT>
void neuralNetwork<ExtraDataT>::build()
{//penser à d'abord clear toute les variables qui seront construites.

    ///assert(computing.try_lock());
    nextCoord.clear();
    previousCoord.clear();
    next.clear();
    previous.clear();
    layersTotalNumberOfNeuron.clear();
    layersCumulatedDimensions.clear();
    vectNeuronsFor.clear();
    vectNeuronsBac.clear();
    vectNeuronsAll.clear();


    for(auto a : neurons)//Construction des dimensions cumulées et totales
    {
        layersTotalNumberOfNeuron[a.first] = totalSize(dimensions[a.first]);
        layersCumulatedDimensions[a.first] = reverseOrderCumulativeSize(dimensions[a.first]);
    }

    if(!parent)//If its an original
    {
        for(auto a : links)
            for(auto b : a.second)
            {
                neuronCoordinate c1 = a.first;
                neuronCoordinate c2 = b.first;
                std::pair <neuron<ExtraDataT>*, double*> toPush;
                toPush = std::make_pair(&neurons[c2.first][c2.second]
                    , &links[c1][c2]);
                next[c1].push_back(toPush);
                nextCoord[c1].push_back(c2);
                toPush.first = &neurons[c1.first][c1.second];
                previous[c2].push_back(toPush);
                previousCoord[c2].push_back(c1);
            }
    }
    else//If its a bond copy
    {
        parent->build();
        nextCoord = parent->nextCoord;
        previousCoord = parent->previousCoord;

        for(auto &neuronsSLink : parent->links)
        {
            for(auto &link : neuronsSLink.second)
            {
                next[neuronsSLink.first].push_back(std::pair<neuron<ExtraDataT>*, double*>(
                    &neurons[link.first.first][link.first.second]
                    , &link.second));
                previous[link.first].push_back(std::pair<neuron<ExtraDataT>*, double*>(
                    &neurons[neuronsSLink.first.first][neuronsSLink.first.second]
                    , &link.second));
            }
        }

    for(auto &a : neurons)
        for(size_t i = 0; i < a.second.size(); i++)
        {
            vectNeuronsAll.emplace_back((&a.second[i]), neuronCoordinate(a.first, i));
            if(!(layersInformationInputOutput[a.first].first))
                vectNeuronsFor.emplace_back(&(a.second[i]), neuronCoordinate(a.first, i));
            if(!(layersInformationInputOutput[a.first].second))
                vectNeuronsBac.emplace_back((&a.second[i]), neuronCoordinate(a.first, i));
        }

    built = 1;
    ///computing.unlock();

    }




}









}


#endif // NEURALNETWORK_H
