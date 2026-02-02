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
#include<random>

using value = std::string;


// push_back di n elementi per worker queue
void operation_in_thread(fdapde::Synchro_queue<value,fdapde::hold_nowait> & q, std::vector<int>::iterator begin,std::vector<int>::iterator end, value el){

    for (auto operation=begin; operation<end; operation++) {
        switch (*operation) {
        case 4:
            q.push_back(el);
            break;
        case 2:
            q.push_front(el);
            break;
        case 3:
            q.pop_back();
            break;
        case 1:
            q.pop_front();
            break;
        }
    }
};


int main(int argc, char** argv){
    int size_coda= std::stoi(argv[1]);

    int n_thread = std::stoi(argv[2]);
    int n_singolo= size_coda / n_thread;

    value el = "ciao";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1,2);

    std::vector<int> operations(size_coda);

    for (int i = 0; i<size_coda;++i) {
        operations[i] = distrib(gen);
        //std::cout << operations[i] << std::endl;
    }


    fdapde::Synchro_queue<value,fdapde::hold_nowait> q(size_coda*2);

    for (int j=0; j<size_coda; j++) {
        q.push_back(el);
    }

    //q.print();

    auto start = std::chrono::high_resolution_clock::now();
    //worker_queue push_back parallelo
    std::vector<std::thread> thread_pool;
    for (int j=0; j<n_thread; j++){
        thread_pool.emplace_back(operation_in_thread,std::ref(q),operations.begin()+j*n_singolo,operations.begin()+(j+1)*n_singolo-1, el);
    }

    for (int k= 0; k<n_thread; k++){
        thread_pool[k].join();
    }
    //q1.print(); //per debug
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    //q.print();

    std::cout<<duration.count()<<",";
    return 0;
}

