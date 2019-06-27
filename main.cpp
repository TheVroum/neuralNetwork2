#include <iostream>












#include "templatelargerscale.h"


///important : créer des spécialisations direcrement
///dans la librairie afin de rendre celle-ci très utilisable


///to use general dropout you can either use normalize neurons callback or intercomputation neuron alteration neural network callback.
///for specialized position dependant dropout, you have to use the above cited neuralnetwork callback





int main(int, char *[])
{
    assert(std::isnan(NAN));//std::numeric_limits::quiet_NaN

    neuralNetwork<int> nn;
    nn.addLayer({2}, 0, 1, 0);
    nn.addLayer({2});
    nn.addLayer({1}, 0, 0, 1, defaultSoftmax<int>);
    nn.connectLayers(0, 1, defaultDense<int>);
    nn.connectLayers(1, 2, defaultDense<int>);
    std::vector<double> v(2, 1);
    layerFeed lf;

    nn.links[neuronCoordinate(0, 0)][neuronCoordinate(1, 1)] = -1;
    nn.links[neuronCoordinate(0, 1)][neuronCoordinate(1, 0)] = -1;

    lf[0] = v;
    auto result = nn.assertion(lf);

    lf[0][0] = 0;
    result = nn.assertion(lf);

    lf[0][1] = 0;
    result = nn.assertion(lf);

    lf[0][0] = 1;
    result = nn.assertion(lf);

    for(auto a : result)
    {
        for(auto b : a.second)
            std::cout << b << std::endl;
        std::cout << std::endl;
    }

    return 0;

}
