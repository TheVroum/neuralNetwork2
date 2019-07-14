#include <iostream>
#include <thread>




#include "neuron.h"
#include "callbackstemplate.h"
#include "templates.h"
#include "neuralnetwork.h"///always include in this order.




///important : créer des spécialisations direcrement
///dans la librairie afin de rendre celle-ci très utilisable


///to use general dropout you can either use normalize neurons callback or intercomputation neuron alteration neural network callback.
///for specialized position dependant dropout, you have to use the above cited neuralnetwork callback




///fix the intercomputationneuronalteration callback who was null at a moment during runtime


///check that all coordinates in callbacks are given through copy and not reference


///remplacer les pointeurs par des biderecitonal links de mon cru


using namespace jo_nn;





void trainingFunc(jo_nn::neuralNetwork<int> *nn)
{
    jo_nn::layerFeed layerBackFeed;
    layerBackFeed[2].resize(1);

    for(size_t i = 0; i < 10000; ++i)
    {
        std::cout  << "\n"  << "\n";
        if(!(i % 4))
            std::cout << "\n" << "\n";

        bool a = i % 2, b = (i / 2) % 2;
        std::cout << a << " ^ " << b << " = " << (a ^ b) << " (expected)" << "\n";

        std::vector<double> v(2);
        v[0] = a;
        v[1] = b;

        jo_nn::layerFeed lf;
        lf[0] = v;

        double res = nn->forCompute(lf)[2][0];
        std::cout << "Result : " << res << ".\tDiff : " << (a ^ b) - res << "." << std::endl;
        layerBackFeed[2][0] = (a ^ b) - res;
        //double diff = abs(layerBackFeed[2][0]);
        nn->backCompute(layerBackFeed, NAN);
    }
}





std::vector <double*> visibilityVector;

int main(int, char *[])//au pire je peux tester directement en mono thread
{
    jo_nn::neuralNetwork<int> nn;
    nn.addLayer({2}, 0, 1, 0, jo_nn::defaultLkRelu<int, 0, 50, 1000>);
    nn.addLayer({2}, 0, 0, 0, jo_nn::defaultLkRelu<int, 0, 50, 1000>);
    nn.addLayer({1}, 0, 0, 1, jo_nn::defaultLkRelu<int, 0, 50, 1000>);
    nn.connectLayers(0, 1, jo_nn::defaultDense<int>);
    nn.connectLayers(1, 2, jo_nn::defaultDense<int>);


    for(auto &a : nn.links)
        for(auto &b : a.second)
            visibilityVector.push_back(&(b.second));

    for(size_t i = 0; i < 4; ++i)
    {
        std::vector<double> v(2);
        v[0] = i % 2;
        v[1] = (i / 2) % 2;

        jo_nn::layerFeed lf;
        lf[0] = v;
        std::cout << nn.assertion(lf)[2][0] << "\t";
    }
    std::cout << std::endl;


    std::vector <jo_nn::neuralNetwork<int>*> trainers;
    trainers = nn.getBindNeuralNetwork(1);

    trainingFunc(trainers[0]);
    std::cout << "\n" << "\n" << "\n" << "\n" << "\n" << "\n";
    for(size_t i = 0; i < 4; ++i)
    {
        std::vector<double> v(2);
        v[0] = i % 2;
        v[1] = (i / 2) % 2;

        jo_nn::layerFeed lf;
        lf[0] = v;
        std::cout << nn.assertion(lf)[2][0] << "\t";
    }
    std::cout << std::endl;

    __asm("int $3");

    return 0;

}
