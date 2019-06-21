#include "sync.h"



//retravailler sync : constructeur deleted et seul constructeur prenant un size_t



sync::sync(sync&&s)
{
    m = s.m;
    v = s.v;
}






void sync::operator()(size_t i)
{
    m[i]->lock();
    v[i]->unlock();
    for(auto a : v)
        a->lock(), a->unlock();

    m[i]->unlock();
    for(auto a : v)
        a->lock(), a->unlock();
    v[i]->lock();
}









sync::~sync()
{
    for(auto a : m)
        delete a;
    for(auto a : v)
        delete a;
}







void sync::operator=(size_t i)///à l'avenir d'avantage utiliser du code répétable
{
    for(auto a : m)
        delete a;
    m.clear();
    for(size_t j = i; j; --j)
        m.push_back(new std::mutex());


    for(auto a : v)
        delete a;
    v.clear();
    for(; i; --i)
        v.push_back(new std::mutex())
        , v.back()->lock();
}







size_t sync::get() const
{
    return m.size();
}












