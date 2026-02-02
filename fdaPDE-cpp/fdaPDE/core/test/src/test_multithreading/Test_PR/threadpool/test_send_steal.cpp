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

#include <fdaPDE/multithreading.h>
#include <atomic>
#include <future>
#include <iostream>

int main(int argc,char** argv){
    int n_worker = std::stoi(argv[1]);
    int n = 10000;

    // round_robin + max_load_stealing
    {
        fdapde::threadpool<fdapde::round_robin_scheduling,fdapde::max_load_stealing> tp(1024,n_worker);
        std::atomic<int> a = 0;
        std::vector<std::future<void>> futs;
        for (int i = 0; i< n; i++){
            futs.emplace_back(tp.send([&](){a++;}));
        }
        for (auto& f:futs){f.get();}
        if(a.load()==n){std::cout<<"round-max funziona\n";}
        else{std::cout<<"round-max NON funziona\n";}
    }

    // round_robin + random_stealing
    {
        fdapde::threadpool<fdapde::round_robin_scheduling,fdapde::random_stealing> tp(1024,n_worker);
        std::atomic<int> a = 0;
        std::vector<std::future<void>> futs;
        for (int i = 0; i< n; i++){
            futs.emplace_back(tp.send([&](){a++;}));
        }
        for (auto& f:futs){f.get();}
        if(a.load()==n){std::cout<<"round-random funziona\n";}
        else{std::cout<<"round-random NON funziona\n";}
    }

    // round_robin + top_half_random_stealing
    {
        fdapde::threadpool<fdapde::round_robin_scheduling,fdapde::top_half_random_stealing> tp(1024,n_worker);
        std::atomic<int> a = 0;
        std::vector<std::future<void>> futs;
        for (int i = 0; i< n; i++){
            futs.emplace_back(tp.send([&](){a++;}));
        }
        for (auto& f:futs){f.get();}
        if(a.load()==n){std::cout<<"round-top funziona\n";}
        else{std::cout<<"round-top NON funziona\n";}
    }

    // least_loaded + max_load_stealing
    {
        fdapde::threadpool<fdapde::least_loaded_scheduling,fdapde::max_load_stealing> tp(1024,n_worker);
        std::atomic<int> a = 0;
        std::vector<std::future<void>> futs;
        for (int i = 0; i< n; i++){
            futs.emplace_back(tp.send([&](){a++;}));
        }
        for (auto& f:futs){f.get();}
        if(a.load()==n){std::cout<<"least-max funziona\n";}
        else{std::cout<<"least-max NON funziona\n";}
    }

    // least_loaded + random_stealing
    {
        fdapde::threadpool<fdapde::least_loaded_scheduling,fdapde::random_stealing> tp(1024,n_worker);
        std::atomic<int> a = 0;
        std::vector<std::future<void>> futs;
        for (int i = 0; i< n; i++){
            futs.emplace_back(tp.send([&](){a++;}));
        }
        for (auto& f:futs){f.get();}
        if(a.load()==n){std::cout<<"least-random funziona\n";}
        else{std::cout<<"least-random NON funziona\n";}
    }

    // least_loaded + top_half_random_stealing
    {
        fdapde::threadpool<fdapde::least_loaded_scheduling,fdapde::top_half_random_stealing> tp(1024,n_worker);
        std::atomic<int> a = 0;
        std::vector<std::future<void>> futs;
        for (int i = 0; i< n; i++){
            futs.emplace_back(tp.send([&](){a++;}));
        }
        for (auto& f:futs){f.get();}
        if(a.load()==n){std::cout<<"least-top funziona\n";}
        else{std::cout<<"least-top NON funziona\n";}
    }

    return 0;
}

