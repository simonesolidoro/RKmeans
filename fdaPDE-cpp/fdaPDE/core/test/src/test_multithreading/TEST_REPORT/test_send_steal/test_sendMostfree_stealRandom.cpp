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

int main(int argc, char** argv){
    
    int n_thread = std::stoi(argv[1]);
    int size_queue = std::stoi(argv[2]);
    int n_cicli_in_job = std::stoi(argv[3]); //lunghezza singolo job 50000
    int n_job = std::stoi(argv[4]); // 10000

    fdapde::Threadpool<fdapde::steal::random> tp(size_queue,n_thread);

    std::vector<std::future<void>> futs;
    futs.reserve(n_job);
    std::atomic<int> count(0);

    auto start2 = std::chrono::high_resolution_clock::now();
    for( int i =0; i<n_job; i++){
        futs.push_back(std::move(tp.send_task_mostfree([&](){
            count++;
            int a= 0;
            for (int j = 0; j<n_cicli_in_job; j++){
                a++;
                a--;
                a++;
                a--;
            }
        })));
    }
    //get dei future per aspettare che lavori siano finiti
    for(int i= 0; i<n_job; i++){
        futs[i].get();
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);  
    if(count.load() == n_job)//per verifica tutti i job eseguiti. non dovrebbe servire 
        std::cout<<duration2.count()<<",";
    return 0;
}