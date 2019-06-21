#include "neuralnetwork.h"






/*

template <typename ExtraDataT>
void neuron::normalize(bool forward)
{
    c_normalize(forward, this);
}
*/






/*


void layer::initialiserTailles()
{
    for(int i = dim.size() - 2; i + 1; i--)
        tailles[i] = tailles[i + 1] * dim[i + 1];
}








size_t layer::coordinatesToIndex(std::vector <size_t> a)
{
    size_t ret = 0;
    for(size_t i = 0; i < a.size(); ++i)
        ret += tailles[i] * a[i];
    return ret;
}

std::vector <size_t> layer::indexToCoordinates(size_t j)
{
    std::vector <size_t> cbdmdpvd;
    for(size_t k = 0; k < tailles.size(); ++k)
    {
        cbdmdpvd.push_back(j / tailles[k]);
        j %= tailles[k];
    }
    return cbdmdpvd;
}


*/




















