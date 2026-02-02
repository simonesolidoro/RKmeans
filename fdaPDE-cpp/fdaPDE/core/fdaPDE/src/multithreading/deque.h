//
// Created by Guido Poletti on 15/04/25.
//

#ifndef DEQUE_H
#define DEQUE_H

#include "header_check.h"

// worker_queue con deque
template <typename T>
class Worker_queue_deque{

    using value_type= T;
private:
    std::deque<std::optional<value_type>> queue_;
    int size_;
    std::mutex m;
public:

    // void per pigrizia tanto solo per test
    void push_front(value_type t){
        std::lock_guard<std::mutex> loc(m);
        queue_.push_back(t);
    }

    std::optional<value_type> pop_front(){
        std::lock_guard<std::mutex> loc(m);
        if(queue_.empty())
            return std::nullopt;
        value_type ret=std::move(queue_.back().value());
        queue_.pop_back();
        return ret;
    }

    //push_back() thread-safe
    void push_back(value_type t){
        std::lock_guard<std::mutex> loc(m);
        queue_.push_front(t);
    }

    //pop_back() thrade-safe
    std::optional<value_type> pop_back(){
        std::lock_guard<std::mutex> loc(m);
        if (queue_.empty())
            return std::nullopt;
        value_type ret=std::move(queue_.front().value());
        queue_.pop_front();
        return ret;
    }

    // wrap of function size() empty() of vector thrade-safe
    int size(){
        std::lock_guard<std::mutex> loc(m);
        return queue_.size();
    }
    bool empty(){//usato in pop durante lock di mutex quindi servirebbe mutex ricorsivo, alternativa togliere il mutex tanto a noi serve solo per test
        return queue_.empty();
    }
    //per debug momentanei
    void print(){
        for (auto i : queue_)
            std::cout<<i.value()<<"  ";
        std::cout<<std::endl;
    }
};

#endif //DEQUE_H
