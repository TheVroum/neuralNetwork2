#ifndef RANDOMNESS_H
#define RANDOMNESS_H


#include <functional>
#include <vector>


//là je fais une version qui se re remplit,
//mais il faudrait en faire une qui a un pool de
//1000 et les utilise en boucle

class normalness
{

public:

    normalness(std::function <int()>);
    double operator()();



private:

    void remplir();


    std::vector <double> v;
    std::function <int()> f;


};







#endif // RANDOMNESS_H
