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









void trainingFunc(jo_nn::neuralNetwork<int> *nn)
{
    jo_nn::layerFeed layerBackFeed;
    layerBackFeed[2].resize(1);

    for(size_t i = 0; i < 1000; ++i)
    {
        bool a = i % 2, b = (i / 2) % 2;

        std::vector<double> v(2);
        v[0] = a;
        v[1] = b;

        jo_nn::layerFeed lf;
        lf[0] = v;

        layerBackFeed[2][0] = (a ^ b) - nn->forCompute(lf)[2][0];
        //double diff = abs(layerBackFeed[2][0]);
        nn->backCompute(layerBackFeed, NAN);
    }
}






int main(int, char *[])//au pire je peux tester directement en mono thread
{
    jo_nn::neuralNetwork<int> nn;
    nn.addLayer({2}, 0, 1, 0);
    nn.addLayer({2});
    nn.addLayer({1}, 0, 0, 1/*, jo_nn::defaultSoftmax<int>*/);
    nn.connectLayers(0, 1, jo_nn::defaultDense<int>);
    nn.connectLayers(1, 2, jo_nn::defaultDense<int>);

    std::vector <jo_nn::neuralNetwork<int>*> result;
    result = nn.getBindNeuralNetwork(4);

    trainingFunc(result[0]);
    //std::thread a(trainingFunc, result[0]);
    std::thread *b = new std::thread(trainingFunc, result[1]);
    std::thread *c = new std::thread(trainingFunc, result[2]);
    std::thread *d = new std::thread(trainingFunc, result[3]);

    /*for(auto a : result)
        delete a;*/

/*
    a.detach();
    b.detach();
    c.detach();
    d.detach();

    a.~thread();
    b.~thread();
    c.~thread();
    d.~thread();
*/

    //a. join();
    b->join();
    c->join();
    d->join();

    for(size_t i = 0; i < 4; ++i)
    {
        std::vector<double> v(2);
        v[0] = i % 2;
        v[1] = (i / 2) % 2;

        jo_nn::layerFeed lf;
        lf[0] = v;
        std::cout << nn.assertion(lf)[2][0];
    }

    __asm("int $3");

    return 0;

}
