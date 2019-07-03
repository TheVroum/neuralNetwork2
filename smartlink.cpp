#include "smartlink.h"




namespace jo_nn
{




/*



template <typename T>
inline smartLink<T>::smartLink(smartLink &s, T* t)
{
    link(s)
}




template <typename T>
inline smartLink<T>::smartLink(smartLink &&s, T* t);




template <typename T>
inline void smartLink<T>::operator=(smartLink&&s, T* t);










template <typename T>
void smartLink<T>::link(smartLink&s, T* t);












template <typename T>
smartLink<T>::smartLink():
other(nullptr),
offset(0)
{
}




template <typename T>
T* smartLink<T>::operator->()
{
    void *ret = reinterpret_cast<void*>(other);
    ret += offset;
    return reinterpret_cast<T*>(ret);
}


*/





smartLinkPtingTo::smartLinkPtingTo():
other(nullptr),
offset(0)
{}





void smartLinkPtingTo::operator=(const smartLinkPtingTo&s)
{
    other = s.other;
    offset = s.offset;
    s.other->others.push_back(this);
}

smartLinkPtingTo::smartLinkPtingTo(const smartLinkPtingTo&s):
other(s.other),
offset(s.offset)
{
    s.other->others.push_back(this);
}


smartLinkPtingTo::smartLinkPtingTo(smartLinkPtingTo&&s):
    other(s.other),
    offset(s.offset)
{
    if(*reinterpret_cast<void**>(&other) != nullptr)
    {
        auto it = std::find(other->others.begin(), other->others.end(), &s);
        if(it != other->others.end())
        {
            *it = this;
        }
        else
        {
            assert(0);//others->push_back(this);
        }
    }
}

smartLinkPtingTo::~smartLinkPtingTo()
{
    if(*reinterpret_cast<void**>(&other) != nullptr)
    {
        auto it = std::find(other->others.begin(), other->others.end(), this);
        assert(it != other->others.end());
        other->others.erase(it);
    }
#ifndef NDEBUG
    other = nullptr;
    offset = 0;
#endif
}





smartLinkPtedTo::~smartLinkPtedTo()
{
    assert(others.empty());
}



}





