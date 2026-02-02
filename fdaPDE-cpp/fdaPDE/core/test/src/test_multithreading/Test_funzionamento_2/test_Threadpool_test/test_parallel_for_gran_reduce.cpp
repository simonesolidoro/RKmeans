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

int main(int argc,char** argv){   
    int n = 10000;
    int granularity = std::stoi(argv[1]); //numero di blocchi in cui dividere range di for
    fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing> tp(64,6);
    std::atomic<int> a=0;
    {
    // reduce max
        a.store(0);
        int end = 10000;
        std::vector<int> v;
        for (int  i =0 ; i<end; i++){
            v.push_back(i+30);
        } 
        v[417]=end+1 ; //minimo 1 in 49
        std::vector<int> vv = {6,3,4,5,2,7,8,9,4,5};
        auto start3 = std::chrono::high_resolution_clock::now();
        std::pair<int,int> min;
        min =tp.parallel_for_reduce<fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing>::reduceOp::max>(0,end,[&](int i){a++;
            return v[i]*35;
        },granularity);
        
        auto end3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
        std::cout<<"par_for_sure_n - incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration3.count()<< " microsecondi con granularity: "<<granularity<<std::endl;
        std::cout<<"massimo trovato: "<<min.first<<"argmax trovato: "<<min.second<<std::endl;
    };
    {
    // reduce min
        a.store(0);
        int end = 10000;
        std::vector<int> v;
        for (int  i =0 ; i<end; i++){
            v.push_back(i+30);
        } 
        v[497]=1 ; //minimo 1 in 497
        std::vector<int> vv = {6,3,4,5,2,7,8,9,4,5};
        auto start3 = std::chrono::high_resolution_clock::now();
        std::pair<int,int> min;
        min =tp.parallel_for_reduce<fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing>::reduceOp::min>(0,end,[&](int i){a++;
            return v[i]*35;
        },granularity);
        
        auto end3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
        std::cout<<"par_for_sure_n - incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration3.count()<< " microsecondi con granularity: "<<granularity<<std::endl;
        std::cout<<"minimo trovato: "<<min.first<<"argmin trovato: "<<min.second<<std::endl;
    };
    // {
    // // reduce sum
    //     a.store(0);
    //     int end = 10000;
    //     std::vector<int> v;
    //     for (int  i =0 ; i<end; i++){
    //         v.push_back(1);
    //     } 
        
    //     int n_it = std::stoi(argv[1]); //numero di blocchi in cui dividere range di for
    //     auto start3 = std::chrono::high_resolution_clock::now();
    //     int dot=0;
    //     dot =tp.parallel_for_reduce_dot(0,end,[&](int i){a++;
    //         return v[i];
    //     },n_it);
        
    //     auto end3 = std::chrono::high_resolution_clock::now();
    //     auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
    //     std::cout<<"par_for_sure_n - incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration3.count()<< " microsecondi con n_it: "<<n_it<<std::endl;
    //     std::cout<<"prodotto trovata (1): "<<dot<<std::endl;

    //     };
    return 0;
};

