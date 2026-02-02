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

int main(int argc,char** argv)
{   
    int n = 10000;
    fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing> tp(2048,8);
    std::atomic<int> a=0;

// parallel_for   granularity
    a.store(0);
    int n_it = std::stoi(argv[1]); //granularity
    auto start3 = std::chrono::high_resolution_clock::now();

    tp.parallel_for(0,1000,[&](int i, int worker_index){a++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));       
    },n_it);
    
    auto end3 = std::chrono::high_resolution_clock::now(); 
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
    std::cout<<"par_for- incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration3.count()<< " microsecondi con granularity: "<<n_it<<std::endl;


}

/*//parallel_for_sure
    int n = 10000;
    fdapde::Threadpool<fdapde::steal::random> tp(64,6);
    std::atomic<int> a=0; //usata per verifica tutti jo vengano eseguiti (a deve arrivare ad n)
    auto start = std::chrono::high_resolution_clock::now();

    tp.parallel_for_sure(0,n,[&](int i){
        a++;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    });
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    std::cout<<"par_for_sure - incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration.count()<< " microsecondi\n";

//for non parallel
    a.store(0);
    auto start2 = std::chrono::high_resolution_clock::now();

    for (int i = 0;i < n;i++){
        a++;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
    std::cout<<"for - incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration2.count()<< " microsecondi\n";
*/