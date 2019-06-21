#ifndef SYNC_H
#define SYNC_H



#include <vector>
#include <mutex>






//retravailler sync : constructeur deleted et seul constructeur prenant un size_t




class sync
{

    std::vector<std::mutex*> m;
    std::vector<std::mutex*> v;

public:

    sync() = default;
    sync(sync&&);
    sync(sync&) = delete;
    void operator =(size_t);
    size_t get() const;

    ~sync();


    void operator()(size_t);


};

#endif // SYNC_H
