#include "randomness.h"

normalness::normalness(std::function <int()> fp):
    f(fp)
{
    remplir();
}



void normalness::remplir()
{
    for(int i = 0; i < 10000; ++i)
    {
        int k = 0;
        for(int j = 0; j < 10000; ++j)
        {
            /*auto e = f();
            e = e % 0x100;
            e = e <= 0x7F;
            k += e;*/
            k += ((int) ((f() % 0x100) <= 0x7F));
        }
        v.push_back((k-((double)5000))/((double)100));
    }
}





double normalness::operator()()
{
    if(v.empty())
        remplir();
    double ret = v.back();
    v.pop_back();
    return ret;
}





