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
    fdapde::threadpool<fdapde::round_robin_scheduling, fdapde::random_stealing> tp(2048,3);
    std::atomic<int> a=0;
    std::vector<int> vect_gran(n/100, 100);
// parallel_for   granularity
    a.store(0);
    int n_it = std::stoi(argv[1]); //op in body function
    auto start3 = std::chrono::high_resolution_clock::now();

    tp.parallel_for(0,n,[&](int i,int index_w, int& n_it){
        
        for (int j =0; j<n_it; j++){
            a.fetch_add(1,std::memory_order_relaxed);
        }
        
    },vect_gran, n_it);
    
    auto end3 = std::chrono::high_resolution_clock::now(); 
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3);  
    std::cout<<"par_for- incrementata a da 0 ad: "<<a.load()<<"  impiegato:"<<duration3.count()<< " microsecondi, doveva  arrivare a : "<<n_it*n<<std::endl;


}
