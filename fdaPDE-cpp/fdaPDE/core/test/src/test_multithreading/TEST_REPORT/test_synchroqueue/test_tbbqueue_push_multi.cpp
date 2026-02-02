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
#include </usr/include/tbb/concurrent_queue.h> 

using value = std::string; 


// push di n elementi per deque 
void push_di_n_elem_d(tbb::concurrent_queue<value> & q,int n, value el){
    for (int j=0; j<n; j++){
        q.push(el);
    }
};

int main(int argc, char** argv){
    int size_coda= std::stoi(argv[1]);

    int n_thread = std::stoi(argv[2]);
    int n_singolo= size_coda / n_thread;

    value el = "ciao";

    tbb::concurrent_queue<value> q;
//push_back() multithreading

    std::vector<std::thread> thread_pool;
    auto start = std::chrono::high_resolution_clock::now();  
    //worker_queue push_back parallelo

    for (int j=0; j<n_thread; j++){
        thread_pool.emplace_back(push_di_n_elem_d,std::ref(q),n_singolo, el);
    }

    for (int k= 0; k<n_thread; k++){
        thread_pool[k].join();
    }
    //q1.print(); //per debug
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);  
    //std::cout<<"push_back in deque di n_elementi: "<<size_coda<<" con n_thread:"<<n_thread<<" impiegato:"<<duration.count()<< " microsecondi\n";
    std::cout<<duration.count()<<",";

    //std::cout<<"queue.unsafe_size(): "<<q.unsafe_size()<<std::endl;
  

}
