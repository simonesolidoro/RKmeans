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

using value = int; 



// push_front di n elementi per deque 
void push_front_di_n_elem_d(Worker_queue_deque<value> & q,int n, value el){
    for (int j=0; j<n; j++){
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        q.push_front(el);
    }
};

void push_front_di_n_elem_s(fdapde::synchro_queue<value,fdapde::relaxed> & q,int n, value el){
    for (int j=0; j<n; j++){
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        q.push_front(el);
    }
};

int main(int argc, char** argv){
    int size_coda= std::stoi(argv[1]);

    int n_thread = std::stoi(argv[2]);
    int n_singolo= size_coda / n_thread;

    Worker_queue_deque<value> q1;
    fdapde::synchro_queue<value,fdapde::relaxed> q2(size_coda);

    std::vector<std::thread> thread_pool;
    std::vector<std::thread> thread_pool2;
    
    //worker_queue push_back parallelo
    for (int j=0; j<n_thread; j++){
        thread_pool.emplace_back(push_front_di_n_elem_d,std::ref(q1),n_singolo, j);
    }

    for (int k= 0; k<n_thread; k++){
        thread_pool[k].join();
    }
    std::cout<<"deque riempita"<<std::endl;
    q1.print(); //per debug
    
    //synchro_queue push_back parallelo
    for (int j=0; j<n_thread; j++){
        thread_pool2.emplace_back(push_front_di_n_elem_s,std::ref(q2),n_singolo, j);
    }

    for (int k= 0; k<n_thread; k++){
        thread_pool2[k].join();
    }
    std::cout<<"synchro queue riempita"<<std::endl;
    q2.print(); //per debug
    
    return 0;
}
