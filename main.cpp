#include "neuralnetwork.h"
#include "templatelargerscale.h"


///important : créer des spécialisations direcrement
///dans la librairie afin de rendre celle-ci très utilisable


///to use general dropout you can either use normalize neurons callback or intercomputation neuron alteration neural network callback.
///for specialized position dependant dropout, you have to use the above cited neuralnetwork callback





int main(int, char *[])
{
    assert(std::isnan(NAN));//std::numeric_limits::quiet_NaN

    neuralNetwork<int> nn;
    nn.addLayer({2}, {0, 0}, 1, 0);
    nn.addLayer({2});
    nn.addLayer({1}, {0, 0}, 0, 1, defaultSoftmax<int>);
    nn.connectLayers();

    return 0;

}
