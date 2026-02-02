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


using value = std::string;

// pop_front di n elementi per worker queue 
void pop_front_di_n_elem(fdapde::Synchro_queue<value,fdapde::hold_nowait> & q,int n){
    for (int j=0; j<n; j++){
        q.pop_front();
    }
};

int main(int argc, char** argv){
    int size_coda= std::stoi(argv[1]);

    int n_thread = std::stoi(argv[2]);
    int n_singolo= size_coda / n_thread;

    fdapde::Synchro_queue<value,fdapde::hold_nowait> q1(size_coda);
    value el = "ciao";
//pop_front() multithreading

    fdapde::Synchro_queue<value,fdapde::hold_nowait> q2(size_coda);

    //popolo
    for (int i=0; i<size_coda; i++){
        q2.push_front(el);
    }
    std::vector<std::thread> thread_pool2;
    auto start2 = std::chrono::high_resolution_clock::now();
    //worker_queue pop_front parallelo

    for (int j=0; j<n_thread; j++){
        thread_pool2.emplace_back(pop_front_di_n_elem,std::ref(q2),n_singolo);
    }

    for (int k= 0; k<n_thread; k++){
        thread_pool2[k].join();
    }    
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
    //std::cout<<"pop_back() in worker_queue di n_elementi: "<<size_coda<<" con n_thread:"<<n_thread<<" impiegato:"<<duration2.count()<< " microsecondi\n";
    std::cout<<duration2.count()<<",";
    return 0;
}

