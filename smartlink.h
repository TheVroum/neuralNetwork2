#ifndef SMARTLINK_H
#define SMARTLINK_H


#include <vector>
#include <cassert>
#include <algorithm>









namespace jo_nn
{


///bidirecitonnal

/*

template <typename T>
class smartLink
{
    smartLink *other;
    int offset;
public:
    smartLink(const smartLink&s) = delete;
    void operator=(const smartLink&s) = delete;


    smartLink();
    T* operator->();

    void link(smartLink&s, T* t);

    inline smartLink(smartLink &s, T* t);
    inline smartLink(smartLink &&s, T* t);
    inline void operator=(smartLink&&s, T* t);
};
*/



///monodirectionnal




class smartLinkPtingTo;



class smartLinkPtedTo
{

friend class smartLinkPtingTo;

    std::vector<smartLinkPtingTo*> others;

public:
    smartLinkPtedTo(const smartLinkPtedTo&s) = delete;
    void operator=(const smartLinkPtedTo&s) = delete;


    smartLinkPtedTo() = default;

    ~smartLinkPtedTo();


    inline smartLinkPtedTo(smartLinkPtedTo&&s);
    inline void operator=(smartLinkPtedTo&&s);
};




class smartLinkPtingTo
{

friend class smartLinkPtedTo;

    smartLinkPtedTo *other;
    int offset;

public:


    smartLinkPtingTo();
    void operator=(const smartLinkPtingTo&s);
    smartLinkPtingTo(const smartLinkPtingTo&s);
    inline void operator=(smartLinkPtingTo&&s);
    smartLinkPtingTo(smartLinkPtingTo&&s);
    ~smartLinkPtingTo();

    template <typename T>
    void link(smartLinkPtedTo&s, T* t);



    /*template <typename T>
    T* operator->();*/


    /*template <typename T>
    T& operator*();*/

    template <typename T>
    T& deref();
};











template <typename T>
void smartLinkPtingTo::link(smartLinkPtedTo&s, T* t)
{
    if(*reinterpret_cast<void**>(&other) != nullptr)
    {
        auto it = std::find(other->others.begin(), other->others.end(), this);
        if(it != other->others.end())
            other->others.erase(it);
    }
    other = &s;
    offset = reinterpret_cast<char*>(t)
        - reinterpret_cast<char*>(other);
}


/*
template <typename T>
T* smartLinkPtingTo::operator->()
{
#ifndef NDEBUG
    assert(other && offset);
#endif
    return reinterpret_cast<T*>(
        reinterpret_cast<char*>(other) + offset);
}
*/





template <typename T>
T& smartLinkPtingTo::deref()
{

#ifndef NDEBUG
    assert(other && offset);
#endif
    return *reinterpret_cast<T*>(
        reinterpret_cast<char*>(other) + offset);
}




inline void smartLinkPtingTo::operator=(smartLinkPtingTo&&s)
{
    other = s.other;
    offset = s.offset;
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






inline smartLinkPtedTo::smartLinkPtedTo(smartLinkPtedTo &&s):
others(s.others)
{
    for(auto &a : others)
        a->other = this;
}


inline void smartLinkPtedTo::operator=(smartLinkPtedTo&&s)
{
    others = s.others;
    for(auto &a : others)
        a->other = this;
}




}




#endif // SMARTLINK_H
