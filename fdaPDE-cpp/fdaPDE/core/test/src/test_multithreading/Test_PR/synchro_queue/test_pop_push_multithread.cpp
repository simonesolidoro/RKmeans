// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include<fdaPDE/multithreading.h>
template<typename sq>
void pushback(sq & q, int size_local){
    for (int i = 0; i<size_local; i++){
        q.push_back(i);
        std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
    }
}
template<typename sq>
void pushfront(sq & q, int size_local){
    for (int i = 0; i<size_local; i++){
        q.push_front(i);
        std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
    }
}
template<typename sq>
void popback(sq & q, int size_local){
    for (int i = 0; i<size_local; i++){
        q.pop_back();
        std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
    }
}
template<typename sq>
void popfront(sq & q, int size_local){
    for (int i = 0; i<size_local; i++){
        q.pop_front();
        std::this_thread::sleep_for(std::chrono::microseconds(1));//cosi che ci sia tempo per far si che si alternino i thread a inserire
    }
}

int main(int argc, char** argv){
    int size_coda=30;
    int n_thread = 3;
    int size_local = 10;
    // using queue = fdapde::synchro_queue<int,fdapde::relaxed>;
    // using queue = fdapde::synchro_queue<int,fdapde::deferred>;
    using queue = fdapde::synchro_queue<int,fdapde::blocking>;
    {
    using queue = fdapde::synchro_queue<int,fdapde::relaxed>;
    std::cout<<"relaxed pop/push front/back multithread"<<std::endl;
        std::cout<<"push_front concorrente:"<<std::endl;
        queue q1(size_coda);
        std::vector<std::thread> threadpool;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(pushfront<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"pop_front concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(popfront<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"push_back concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(pushback<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"pop_back concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(popback<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();
    }
    std::cout<<std::endl;
    {
    using queue = fdapde::synchro_queue<int,fdapde::deferred>;
    std::cout<<"deferred pop/push front/back multithread"<<std::endl;
        std::cout<<"push_front concorrente:"<<std::endl;
        queue q1(size_coda);
        std::vector<std::thread> threadpool;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(pushfront<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"pop_front concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(popfront<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"push_back concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(pushback<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"pop_back concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(popback<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();
    }
    std::cout<<std::endl;
    {
    using queue = fdapde::synchro_queue<int,fdapde::blocking>;
    std::cout<<"blocking pop/push front/back multithread"<<std::endl;
        std::cout<<"push_front concorrente:"<<std::endl;
        queue q1(size_coda);
        std::vector<std::thread> threadpool;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(pushfront<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"pop_front concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(popfront<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"push_back concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(pushback<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();

        threadpool.clear();
        std::cout<<"pop_back concorrente:"<<std::endl;
        for(int i = 0; i<n_thread; i++){
            threadpool.emplace_back(popback<queue>,std::ref(q1),size_local);
        }
        for(int i = 0; i<n_thread; i++){
            threadpool[i].join();
        }
        q1.print();
    }
}
